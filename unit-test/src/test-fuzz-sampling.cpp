#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// sampler/pdf verification for the ported zeltner ltc fuzz (sheen) lobe
// (openPbrFuzzSampleWo / openPbrFuzzPdf), following the openpbr port
// workflow: golden/degenerate closed-form values, dense NaN sweeps,
// chi-square sampler-vs-pdf statistics (modeled on cull's bsdf-verify
// harness), independent materialx zeltner-sheen cross-checks (both the
// algebraic pdf identity and the matched-xi sample path), pdf-integrates-
// to-1 quadrature, rotation invariance, and heatmap visualizations.
// ---------------------------------------------------------------------------

namespace {

// must match util-sampling-bins.glsl
constexpr u32 kThetaBins = 40u;
constexpr u32 kPhiBins = 80u;
constexpr u32 kBinCount = kThetaBins * kPhiBins;

// must match sampling_histogram.comp / sampling_expected.comp
constexpr u32 kModeFuzzWo = 3u;

struct FlatPush {
	u64 inVa;
	u64 outVa;
	u32 count;
};

// generic runner for the flat f32-in / f32-out evaluate/crosscheck shaders
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

// layout must match the scalar push block in fuzz_sample_sweep.comp
struct SweepPush {
	u64 outVa;
	u32 width;
	u32 height;
	f32 fuzzB;
};

// 4 floats per texel: nonFiniteCount, maxUnitLengthError, minPdf, maxPdf
std::vector<f32> run_sweep(
	u32 const width, u32 const height, f32 const fuzzB
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fuzz_sample_sweep.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto buf = vkof::buffer_create({
		.byteCount = (u64)width * height * 4u * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	SweepPush const push {
		.outVa = vkof::buffer_virtual_address(buf),
		.width = width,
		.height = height,
		.fuzzB = fuzzB,
	};
	test::dispatch(pl, push, (width + 7u) / 8u, (height + 7u) / 8u);

	auto out = test::readback<f32>(buf, 0, width * height * 4u);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

// layout must match the scalar push block in sampling_histogram.comp /
// sampling_expected.comp; alpha.xy repurposed as (fuzzA, fuzzB)
struct SamplingStatsPush {
	u64 bufferVa;
	u32 mode;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 wi;
	f32v2 fuzzAb;
	// transmission-only; unused by this mode
	f32v2 aniso;
};
static_assert(sizeof(SamplingStatsPush) == 48);

std::vector<u32> run_histogram(
	f32v3 const wi, f32v2 const fuzzAb, u32 const sampleCount, u32 const seedSalt
) {
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
		.mode = kModeFuzzWo,
		.sampleCount = sampleCount,
		.seedSalt = seedSalt,
		.wi = wi,
		.fuzzAb = fuzzAb,
		.aniso = { 0.0f, 0.0f },
	};
	static constexpr u32 kLocalSize = 256;
	test::dispatch(pl, push, (sampleCount + kLocalSize - 1) / kLocalSize);

	auto out = test::readback<u32>(buf, 0, slotCount);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

std::vector<f32> run_expected(
	f32v3 const wi, f32v2 const fuzzAb, u32 const sampleCount
) {
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
		.mode = kModeFuzzWo,
		.sampleCount = sampleCount,
		.seedSalt = 0u,
		.wi = wi,
		.fuzzAb = fuzzAb,
		.aniso = { 0.0f, 0.0f },
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

// pearson chi-square with E < 5 pooling and a wilson-hilferty p-value (same
// construction as test-microfacet-sampling.cpp / test-diffuse-sampling.cpp)
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

void check_chi_square(
	f32v3 const wi, f32v2 const fuzzAb, u32 const seedSalt,
	char const * const residualPngPath = nullptr
) {
	constexpr u32 kSampleCount = 1u << 20;
	auto const observed = run_histogram(wi, fuzzAb, kSampleCount, seedSalt);
	auto const expected = run_expected(wi, fuzzAb, kSampleCount);

	CAPTURE(wi.z);
	CAPTURE(fuzzAb.x);
	CAPTURE(fuzzAb.y);

	CHECK(observed[kBinCount] == 0u);

	auto const cs = chi_square(observed, expected);
	CAPTURE(cs.statistic);
	CAPTURE(cs.dof);
	CAPTURE(cs.pooledObserved);
	CAPTURE(cs.pooledExpected);
	CHECK(cs.observedTotal == kSampleCount);
	CHECK(
		cs.expectedTotal
		== doctest::Approx((f64)kSampleCount).epsilon(0.03)
	);
	CHECK(cs.pValue > 1e-4);
	CHECK(cs.pooledObserved <= cs.pooledExpected + 64.0);

	if (residualPngPath != nullptr) {
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
			r, g, b, kPhiBins, kThetaBins, residualPngPath, &nanCount
		);
		CHECK(ok);
		CHECK(nanCount == 0);
	}
}

// deterministic lcg in [0, 1), matching the other test files
struct Rng {
	u32 seed;
	f32 next() {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	}
};

f32v3 random_unit(Rng & rng) {
	f32 const z = rng.next() * 2.0f - 1.0f;
	f32 const phi = rng.next() * 6.2831853f;
	f32 const s = std::sqrt(std::fmax(0.0f, 1.0f - z * z));
	return { s * std::cos(phi), s * std::sin(phi), z };
}

f32v3 rotate_axis_angle(f32v3 const v, f32v3 const axis, f32 const angle) {
	// rodrigues rotation; axis must be unit length
	f32 const c = std::cos(angle);
	f32 const s = std::sin(angle);
	f32v3 const cx {
		axis.y * v.z - axis.z * v.y,
		axis.z * v.x - axis.x * v.z,
		axis.x * v.y - axis.y * v.x,
	};
	f32 const d = axis.x * v.x + axis.y * v.y + axis.z * v.z;
	return {
		v.x * c + cx.x * s + axis.x * d * (1.0f - c),
		v.y * c + cx.y * s + axis.y * d * (1.0f - c),
		v.z * c + cx.z * s + axis.z * d * (1.0f - c),
	};
}

void push10(
	std::vector<f32> & flat, f32v3 const nor, f32v3 const wi,
	f32 const fuzzA, f32 const fuzzB, f32v2 const xi
) {
	flat.push_back(nor.x); flat.push_back(nor.y); flat.push_back(nor.z);
	flat.push_back(wi.x); flat.push_back(wi.y); flat.push_back(wi.z);
	flat.push_back(fuzzA); flat.push_back(fuzzB);
	flat.push_back(xi.x); flat.push_back(xi.y);
}

void push11(
	std::vector<f32> & flat, f32v3 const nor, f32v3 const wi, f32v3 const wo,
	f32 const fuzzA, f32 const fuzzB
) {
	flat.push_back(nor.x); flat.push_back(nor.y); flat.push_back(nor.z);
	flat.push_back(wi.x); flat.push_back(wi.y); flat.push_back(wi.z);
	flat.push_back(wo.x); flat.push_back(wo.y); flat.push_back(wo.z);
	flat.push_back(fuzzA); flat.push_back(fuzzB);
}

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("fuzz sampler: unit length, upper hemisphere, pdf > 0 at own samples (fuzz)") {
	std::vector<f32> flat;
	Rng rng { 2026u };
	constexpr u32 kCount = 8192u;
	std::vector<f32v3> nors;
	for (u32 i = 0; i < kCount; ++i) {
		f32v3 const nor = random_unit(rng);
		f32 const cosTheta = -1.0f + 2.0f * rng.next();
		f32 const sinTheta = std::sqrt(std::fmax(0.0f, 1.0f - cosTheta * cosTheta));
		f32 const phiI = rng.next() * 6.2831853f;
		// wi expressed relative to nor via an arbitrary tangent frame is
		// awkward to build host-side without utilCalculateXy; instead
		// rotate a canonical-frame wi onto the random nor
		f32v3 const wiLocal { sinTheta * std::cos(phiI), sinTheta * std::sin(phiI), cosTheta };
		f32v3 const axis = random_unit(rng);
		f32 const angle = rng.next() * 6.2831853f;
		f32v3 const nor2 = rotate_axis_angle({ 0.0f, 0.0f, 1.0f }, axis, angle);
		f32v3 const wi = rotate_axis_angle(wiLocal, axis, angle);
		f32 const fuzzA = 1e-5f + rng.next() * 3.0f;
		f32 const fuzzB = -2.0f + rng.next() * 4.0f;
		nors.push_back(nor2);
		push10(flat, nor2, wi, fuzzA, fuzzB, { rng.next(), rng.next() });
	}
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "fuzz_sample_evaluate.comp", flat, 10u, 4u
	);
	u32 lenFailures = 0u;
	u32 horizonFailures = 0u;
	u32 pdfFailures = 0u;
	for (u32 i = 0; i < kCount; ++i) {
		f32 const x = out[i*4+0], y = out[i*4+1], z = out[i*4+2];
		f32 const pdf = out[i*4+3];
		CAPTURE(i);
		bool const finite = (
			std::isfinite(x) && std::isfinite(y) && std::isfinite(z)
			&& std::isfinite(pdf)
		);
		CHECK(finite);
		if (!finite) { continue; }
		f32 const len = std::sqrt(x*x + y*y + z*z);
		if (std::fabs(len - 1.0f) > 1e-3f) { lenFailures++; }
		f32 const dotNorWo = nors[i].x*x + nors[i].y*y + nors[i].z*z;
		// the ltc-pushed cosine sample always lands in the nor-upper
		// hemisphere (t.z = dLocal.z = sqrt(xi.x) >= 0 is preserved
		// through the orthonormal world-frame build and normalize, see
		// the fuzz_pdf_integrate.comp doc comment)
		if (dotNorWo < -1e-3f) { horizonFailures++; }
		if (!(pdf > 0.0f)) { pdfFailures++; }
	}
	CHECK(lenFailures == 0u);
	CHECK(horizonFailures == 0u);
	CHECK(pdfFailures == 0u);
}

TEST_CASE("fuzz sampler: golden -- xi.x = 0 always samples exactly grazing") {
	// dLocal.z = sqrt(xi.x); at xi.x = 0, tLocal.z = dLocal.z = 0
	// (the matrix's third row is identity), so dot(nor, wo) = tLocal.z /
	// |tLocal| = 0 exactly -- independent of wi, xi.y, fuzzB, and fuzzA
	// (as long as fuzzA is not floored to 0, in which case tLocal.x, y
	// blow up but z stays 0 and the ratio is still 0/inf = 0)
	std::vector<f32> flat;
	Rng rng { 314u };
	constexpr u32 kCount = 512u;
	for (u32 i = 0; i < kCount; ++i) {
		f32v3 const wi = random_unit(rng);
		f32 const fuzzA = 1e-5f + rng.next() * 3.0f;
		f32 const fuzzB = -2.0f + rng.next() * 4.0f;
		push10(
			flat, { 0.0f, 0.0f, 1.0f }, wi, fuzzA, fuzzB, { 0.0f, rng.next() }
		);
	}
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "fuzz_sample_evaluate.comp", flat, 10u, 4u
	);
	for (u32 i = 0; i < kCount; ++i) {
		CAPTURE(i);
		CHECK(std::isfinite(out[i*4+2]));
		CHECK(out[i*4+2] == doctest::Approx(0.0).epsilon(1e-4));
	}
}

TEST_CASE("fuzz sampler: golden -- xi.x = 1 samples the closed-form apex cosine") {
	// dLocal = (0, 0, 1) exactly, so tLocal = (-fuzzB/safeA, 0, 1)
	// independent of wi and xi.y; dot(nor, wo) = 1 / sqrt(1 +
	// (fuzzB/safeA)^2). restricted to fuzzA >= the 1e-5 safeA floor so
	// safeA == fuzzA and the closed form applies directly.
	std::vector<f32> flat;
	std::vector<f32> expectedCosine;
	Rng rng { 2718u };
	constexpr u32 kCount = 256u;
	for (u32 i = 0; i < kCount; ++i) {
		f32v3 const wi = random_unit(rng);
		f32 const fuzzA = 1e-5f + rng.next() * 3.0f;
		f32 const fuzzB = -2.0f + rng.next() * 4.0f;
		f32 const r = fuzzB / fuzzA;
		expectedCosine.push_back(1.0f / std::sqrt(1.0f + r * r));
		push10(
			flat, { 0.0f, 0.0f, 1.0f }, wi, fuzzA, fuzzB,
			{ 1.0f, rng.next() }
		);
	}
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "fuzz_sample_evaluate.comp", flat, 10u, 4u
	);
	for (u32 i = 0; i < kCount; ++i) {
		CAPTURE(i);
		CHECK(std::isfinite(out[i*4+2]));
		CHECK(out[i*4+2] == doctest::Approx(expectedCosine[i]).epsilon(2e-4));
	}
}

TEST_CASE("fuzz sampler: degenerate wi parallel/antiparallel to nor stay finite") {
	// wi == nor: utilFuzzFrame's projection collapses (tRaw = 0), falls
	// back to utilCalculateXy. wi == -nor: same fallback (tRaw = wi -
	// nor*dot(nor,wi) = -nor - nor*(-1) = 0 too)
	std::vector<f32> flat;
	std::vector<f32> const fuzzAs = { 1e-5f, 0.0f, 0.3f, 1.0f, 3.0f };
	std::vector<f32> const fuzzBs = { -1.5f, 0.0f, 1.5f };
	std::vector<f32> const xixs = { 0.0f, 0.5f, 0.99999988f };
	for (f32 const fuzzA : fuzzAs) {
		for (f32 const fuzzB : fuzzBs) {
			for (f32 const xix : xixs) {
				push10(
					flat, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f },
					fuzzA, fuzzB, { xix, 0.3f }
				);
				push10(
					flat, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, -1.0f },
					fuzzA, fuzzB, { xix, 0.3f }
				);
			}
		}
	}
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "fuzz_sample_evaluate.comp", flat, 10u, 4u
	);
	for (u32 i = 0; i < out.size() / 4u; ++i) {
		CAPTURE(i);
		CHECK(std::isfinite(out[i*4+0]));
		CHECK(std::isfinite(out[i*4+1]));
		CHECK(std::isfinite(out[i*4+2]));
		CHECK(std::isfinite(out[i*4+3]));
		f32 const len = std::sqrt(
			out[i*4+0]*out[i*4+0] + out[i*4+1]*out[i*4+1] + out[i*4+2]*out[i*4+2]
		);
		CHECK(len == doctest::Approx(1.0).epsilon(1e-3));
	}
}

TEST_CASE("fuzz pdf: rotation invariance") {
	// pdf(nor, wi, wo, fuzzA, fuzzB) must not depend on the frame's world
	// orientation: pdf(R nor, R wi, R wo) == pdf(nor, wi, wo) for any
	// rotation R
	std::vector<f32> flatBase;
	std::vector<f32> flatRotated;
	Rng rng { 6060u };
	constexpr u32 kCount = 2048u;
	for (u32 i = 0; i < kCount; ++i) {
		f32v3 const wi = random_unit(rng);
		f32v3 const wo = random_unit(rng);
		f32 const fuzzA = 1e-5f + rng.next() * 3.0f;
		f32 const fuzzB = -2.0f + rng.next() * 4.0f;
		f32v3 const axis = random_unit(rng);
		f32 const angle = rng.next() * 6.2831853f;
		f32v3 const nor { 0.0f, 0.0f, 1.0f };
		push11(flatBase, nor, wi, wo, fuzzA, fuzzB);
		push11(
			flatRotated,
			rotate_axis_angle(nor, axis, angle),
			rotate_axis_angle(wi, axis, angle),
			rotate_axis_angle(wo, axis, angle),
			fuzzA, fuzzB
		);
	}
	auto const base = run_flat_shader(
		TEST_SHADER_DIR "fuzz_pdf_evaluate.comp", flatBase, 11u, 1u
	);
	auto const rotated = run_flat_shader(
		TEST_SHADER_DIR "fuzz_pdf_evaluate.comp", flatRotated, 11u, 1u
	);
	u32 mismatches = 0u;
	for (u32 i = 0; i < kCount; ++i) {
		CAPTURE(i);
		CAPTURE(base[i]);
		CAPTURE(rotated[i]);
		CHECK(std::isfinite(base[i]));
		CHECK(std::isfinite(rotated[i]));
		bool const match = (
			std::fabs(base[i] - rotated[i])
			<= 2e-3f * std::fmax(std::fabs(base[i]), std::fabs(rotated[i])) + 1e-5f
		);
		if (!match) { mismatches++; }
	}
	CHECK(mismatches == 0u);
}

TEST_CASE("fuzz sampler: NaN sweep heatmap (fuzzB negative/zero/positive)") {
	constexpr u32 kSize = 256u;
	struct SweepConfig { f32 fuzzB; char const * path; };
	std::vector<SweepConfig> const configs = {
		{ -2.0f, FUZZ_OUTPUT_DIR "fuzz_nan_sweep_bneg.png" },
		{ 0.0f, FUZZ_OUTPUT_DIR "fuzz_nan_sweep_bzero.png" },
		{ 1.5f, FUZZ_OUTPUT_DIR "fuzz_nan_sweep_bpos.png" },
	};
	for (auto const & cfg : configs) {
		auto const out = run_sweep(kSize, kSize, cfg.fuzzB);

		std::vector<f32> r(kSize * kSize), g(kSize * kSize), b(kSize * kSize);
		u32 nonFinite = 0u;
		f32 maxLenErr = 0.0f;
		u32 pdfNegativeTexels = 0u;
		for (u32 y = 0; y < kSize; ++y) {
			for (u32 x = 0; x < kSize; ++x) {
				u32 const i = y * kSize + x;
				f32 const lenErr = out[i*4+1];
				f32 const minPdf = out[i*4+2];
				f32 const maxPdf = out[i*4+3];
				if (out[i*4+0] > 0.0f) {
					nonFinite++;
					r[i] = g[i] = b[i] = std::nanf("");
					continue;
				}
				maxLenErr = std::fmax(maxLenErr, lenErr);
				// xi.x = 0 is in this grid (xj = 0) and is a legitimate
				// exact-zero-density grazing sample (see the "golden --
				// xi.x = 0" test), so pdf == 0 is expected at every texel;
				// only a strictly negative pdf indicates a real bug
				if (minPdf < 0.0f) { pdfNegativeTexels++; }
				f32 const v = maxPdf / (maxPdf + 1.0f);
				r[i] = v; g[i] = v; b[i] = v;
			}
		}
		u32 nanCount = 0;
		bool const ok = test::write_heatmap_png(
			r, g, b, kSize, kSize, cfg.path, &nanCount
		);
		CHECK(ok);
		CAPTURE(cfg.fuzzB);
		CHECK(nonFinite == 0u);
		CHECK(maxLenErr < 1e-3f);
		CHECK(pdfNegativeTexels == 0u);
	}
}

TEST_CASE("fuzz pdf: integrates to ~1 over the sphere") {
	std::vector<f32> flat;
	struct Config { f32v3 wi; f32 fuzzA; f32 fuzzB; f32 tolerance; };
	std::vector<Config> configs;
	std::vector<f32> const cosThetas = { 1.0f, 0.7f, 0.3f, 0.05f, -0.4f };
	std::vector<f32> const fuzzAs = { 0.15f, 0.5f, 1.0f, 2.0f };
	for (f32 const ct : cosThetas) {
		f32 const st = std::sqrt(std::fmax(0.0f, 1.0f - ct * ct));
		for (f32 const fuzzA : fuzzAs) {
			configs.push_back({
				{ st * 0.8f, st * 0.6f, ct }, fuzzA, -0.6f, 0.02f
			});
			configs.push_back({
				{ st * 0.8f, st * 0.6f, ct }, fuzzA, 0.4f, 0.02f
			});
		}
		// near-delta fuzzA: the ltc lobe compresses into a small cap,
		// under-resolved by the fixed 1024x512 quadrature grid
		configs.push_back({ { st * 0.8f, st * 0.6f, ct }, 0.02f, 0.0f, 0.08f });
	}
	for (auto const & c : configs) {
		flat.push_back(0.0f); flat.push_back(0.0f); flat.push_back(1.0f);
		flat.push_back(c.wi.x); flat.push_back(c.wi.y); flat.push_back(c.wi.z);
		flat.push_back(c.fuzzA); flat.push_back(c.fuzzB);
	}
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "fuzz_pdf_integrate.comp", flat, 8u, 1u
	);
	for (u32 i = 0; i < configs.size(); ++i) {
		CAPTURE(i);
		CAPTURE(configs[i].wi.z);
		CAPTURE(configs[i].fuzzA);
		CAPTURE(configs[i].fuzzB);
		CHECK(std::isfinite(out[i]));
		CHECK(out[i] == doctest::Approx(1.0).epsilon(configs[i].tolerance));
	}
}

TEST_CASE("fuzz pdf: sphere-integral heatmap (incidence x fuzzA)") {
	constexpr u32 kWidth = 48u;
	constexpr u32 kHeight = 48u;
	std::vector<f32> flat;
	for (u32 y = 0; y < kHeight; ++y) {
		f32 const fuzzA = 0.1f + (2.5f - 0.1f) * (f32)y / (f32)(kHeight - 1);
		for (u32 x = 0; x < kWidth; ++x) {
			f32 const ct = -0.99f + 1.98f * (f32)x / (f32)(kWidth - 1);
			f32 const st = std::sqrt(std::fmax(0.0f, 1.0f - ct * ct));
			flat.push_back(0.0f); flat.push_back(0.0f); flat.push_back(1.0f);
			flat.push_back(st); flat.push_back(0.0f); flat.push_back(ct);
			flat.push_back(fuzzA); flat.push_back(0.0f);
		}
	}
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "fuzz_pdf_integrate.comp", flat, 8u, 1u
	);
	std::vector<f32> img(out.size());
	f64 maxDeviation = 0.0;
	for (u32 y = 0; y < kHeight; ++y) {
		f32 const fuzzA = 0.1f + (2.5f - 0.1f) * (f32)y / (f32)(kHeight - 1);
		for (u32 x = 0; x < kWidth; ++x) {
			u32 const i = y * kWidth + x;
			img[i] = std::isfinite(out[i]) ? out[i] * 0.5f : std::nanf("");
			if (!std::isfinite(out[i])) { continue; }
			if (fuzzA >= 0.2f) {
				maxDeviation = std::fmax(maxDeviation, std::fabs((f64)out[i] - 1.0));
			}
		}
	}
	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		img, img, img, kWidth, kHeight,
		FUZZ_OUTPUT_DIR "fuzz_pdf_integral_heatmap.png", &nanCount
	);
	CHECK(ok);
	CAPTURE(maxDeviation);
	CHECK(nanCount == 0u);
	CHECK(maxDeviation < 0.06);
}

TEST_CASE("fuzz pdf vs materialx zeltner sheen brdf: algebraic cross-check") {
	// mx_zeltner_sheen_brdf and openPbrFuzzPdf are the identical LTC
	// density derivation; feeding materialx's own aInv/bInv into
	// openPbrFuzzPdf must reproduce mx_zeltner_sheen_brdf's return to fp
	// precision at any (wi, wo, roughness), independent of any sampler.
	//
	// two domain caveats, both matching conventions already established by
	// the diffuse/ggx crosschecks rather than being fuzz-specific bugs:
	// (1) mx_cosine_hemisphere_PDF clamps its cosine to >= 0, so
	// mx_zeltner_sheen_brdf returns exactly 0 for a below-nor-horizon wo;
	// openPbrFuzzPdf has no such clamp (same convention as
	// openPbrGlossyDiffusePdf -- see sampling_expected.comp), so the two
	// are compared via max(mine, 0) rather than raw equality.
	// (2) mx_orthonormal_basis_ltc builds its tangent from a NdotV that the
	// caller has already clamped to [0, 1] (`V - N*NdotV`), so for a
	// below-horizon wi it uses V verbatim instead of V's tangential
	// projection -- a materialx-side convention (its sheen bsdf is never
	// evaluated below the local horizon in production) that ours does not
	// share (utilFuzzFrame always projects). wi is restricted to the upper
	// hemisphere here so both sides are exercising the same basis.
	std::vector<f32> flat;
	Rng rng { 909090u };
	constexpr u32 kCount = 4096u;
	for (u32 i = 0; i < kCount; ++i) {
		// wi restricted to the upper hemisphere (cosTheta in [0.05, 1]),
		// see the domain caveat above
		f32 const cosTheta = 0.05f + rng.next() * 0.95f;
		f32 const sinTheta = std::sqrt(std::fmax(0.0f, 1.0f - cosTheta * cosTheta));
		f32 const phi = rng.next() * 6.2831853f;
		f32v3 const wi {
			sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta
		};
		f32v3 const wo = random_unit(rng);
		f32 const roughness = rng.next();
		flat.push_back(wi.x); flat.push_back(wi.y); flat.push_back(wi.z);
		flat.push_back(wo.x); flat.push_back(wo.y); flat.push_back(wo.z);
		flat.push_back(roughness);
	}
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "fuzz_pdf_crosscheck.comp", flat, 7u, 2u
	);
	u32 mismatches = 0u;
	for (u32 i = 0; i < kCount; ++i) {
		CAPTURE(i);
		f32 const mine = std::fmax(out[i*2+0], 0.0f);
		f32 const ref = out[i*2+1];
		CHECK(std::isfinite(out[i*2+0]));
		CHECK(std::isfinite(ref));
		bool const match = (
			std::fabs(mine - ref) <= 1e-4f * std::fmax(std::fabs(ref), 1.0f)
		);
		if (!match) { mismatches++; }
	}
	CHECK(mismatches == 0u);
}

TEST_CASE("fuzz sampler vs materialx zeltner importance sample: matched-xi cross-check") {
	// wi restricted to the upper hemisphere: mx_orthonormal_basis_ltc
	// builds its tangent from a NdotV the caller has already clamped to
	// [0, 1] (V - N*NdotV), so for a below-horizon wi it uses V verbatim
	// instead of V's tangential projection -- a materialx-side convention
	// (never evaluated below the local horizon in production) that ours
	// does not share (utilFuzzFrame always projects), same domain caveat
	// as the pdf algebraic cross-check above
	std::vector<f32> flat;
	Rng rng { 111213u };
	constexpr u32 kCount = 4096u;
	for (u32 i = 0; i < kCount; ++i) {
		f32 const cosTheta = 0.05f + rng.next() * 0.95f;
		f32 const sinTheta = std::sqrt(std::fmax(0.0f, 1.0f - cosTheta * cosTheta));
		f32 const phi = rng.next() * 6.2831853f;
		f32v3 const wi {
			sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta
		};
		f32 const roughness = rng.next();
		flat.push_back(wi.x); flat.push_back(wi.y); flat.push_back(wi.z);
		flat.push_back(roughness);
		flat.push_back(rng.next()); flat.push_back(rng.next());
	}
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "fuzz_sample_crosscheck.comp", flat, 6u, 8u
	);
	u32 dirMismatches = 0u;
	u32 pdfMismatches = 0u;
	for (u32 i = 0; i < kCount; ++i) {
		CAPTURE(i);
		f32v3 const mine { out[i*8+0], out[i*8+1], out[i*8+2] };
		f32v3 const ref { out[i*8+3], out[i*8+4], out[i*8+5] };
		f32 const minePdf = out[i*8+6];
		f32 const refPdf = out[i*8+7];
		CHECK(std::isfinite(mine.x));
		CHECK(std::isfinite(ref.x));
		bool const dirOk = (
			std::fabs(mine.x - ref.x) < 3e-4f
			&& std::fabs(mine.y - ref.y) < 3e-4f
			&& std::fabs(mine.z - ref.z) < 3e-4f
		);
		if (!dirOk) { dirMismatches++; }
		bool const pdfOk = (
			std::fabs(minePdf - refPdf) <= 2e-3f * std::fmax(std::fabs(refPdf), 1.0f)
		);
		if (!pdfOk) { pdfMismatches++; }
	}
	CHECK(dirMismatches == 0u);
	CHECK(pdfMismatches == 0u);
}

TEST_CASE("fuzz sampler vs pdf: chi-square") {
	u32 salt = 700u;
	std::vector<f32> const fuzzAs = { 0.3f, 0.8f, 1.5f };
	std::vector<f32> const cosThetas = { 1.0f, 0.6f, 0.2f, -0.3f };
	for (f32 const fuzzA : fuzzAs) {
		for (f32 const ct : cosThetas) {
			f32 const st = std::sqrt(std::fmax(0.0f, 1.0f - ct * ct));
			check_chi_square(
				{ st, 0.0f, ct }, { fuzzA, -0.3f }, salt++
			);
		}
	}
}

TEST_CASE("fuzz sampler vs pdf: chi-square residual heatmap") {
	check_chi_square(
		{ 0.6f, 0.0f, 0.8f }, { 0.6f, -0.4f }, 800u,
		FUZZ_OUTPUT_DIR "fuzz_chi2_residual.png"
	);
	check_chi_square(
		{ 0.6f, 0.0f, 0.8f }, { 0.6f, 0.5f }, 801u,
		FUZZ_OUTPUT_DIR "fuzz_chi2_residual_bpos.png"
	);
}

} // TEST_SUITE("[headless]")
