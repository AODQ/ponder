#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// openpbr thin-walled mode replaces the subsurface random walk with a pair of
// diffuse lobes (openPbrThinWalledSubsurfaceEvaluateF / Pdf / SampleWo in
// util-material-openpbr-diffuse.glsl), splitting S = subsurface_color between
// a diffuse reflection and a diffuse transmission lobe by
// g = subsurface_scatter_anisotropy:
//
//   f^R = 1/2 S (1 - g) f_+ ,  f^T = 1/2 S (1 + g) f_-  (spec eq
//   thin_wall_subsurface), f_\pm albedo-1 EON lobes shaped by
//   base_diffuse_roughness
//
// following the openpbr port workflow: closed-form/materialx crosscheck of
// the lobe shape, the spec's own energy statement E_R + E_T = S as a furnace
// check, pdf sphere normalization, sampler-vs-pdf chi-square, a two-sidedness
// check (the sheet must look identical from either side), and heatmaps.
// ---------------------------------------------------------------------------

namespace {

// must match util-sampling-bins.glsl
constexpr u32 kThetaBins = 40u;
constexpr u32 kPhiBins = 80u;
constexpr u32 kBinCount = kThetaBins * kPhiBins;

constexpr f64 kPi = 3.14159265358979323846;

u32 groups_for(u32 const count, u32 const localSize) {
	return (count + localSize - 1u) / localSize;
}

// -----------------------------------------------------------------------------
// thin_walled_subsurface_evaluate.comp runner
// -----------------------------------------------------------------------------

struct EvaluateOut {
	f32v3 f;
	f32 pdf;
	f32v3 reference;
};

void push_evaluate_sample(
	std::vector<f32> & flat,
	f32v3 const nor, f32v3 const wi, f32v3 const wo,
	f32 const sigma, f32 const anisotropy, f32v3 const subsurfaceColor
) {
	flat.emplace_back(nor.x);
	flat.emplace_back(nor.y);
	flat.emplace_back(nor.z);
	flat.emplace_back(wi.x);
	flat.emplace_back(wi.y);
	flat.emplace_back(wi.z);
	flat.emplace_back(wo.x);
	flat.emplace_back(wo.y);
	flat.emplace_back(wo.z);
	flat.emplace_back(sigma);
	flat.emplace_back(anisotropy);
	flat.emplace_back(subsurfaceColor.x);
	flat.emplace_back(subsurfaceColor.y);
	flat.emplace_back(subsurfaceColor.z);
}

std::vector<EvaluateOut> run_evaluate(std::vector<f32> const & samplesFlat) {
	u32 const count = (u32)samplesFlat.size() / 14u;
	REQUIRE(samplesFlat.size() == (size_t)count * 14u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "thin_walled_subsurface_evaluate.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto inBuf = vkof::buffer_create({
		.byteCount = samplesFlat.size() * sizeof(f32),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = inBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(samplesFlat.data()),
			samplesFlat.size() * sizeof(f32)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = count * 8u * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push { u64 inVa; u64 outVa; u32 count; };
	Push const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count, 64u));

	auto raw = test::readback<f32>(outBuf, 0, count * 8u);
	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);

	std::vector<EvaluateOut> out(count);
	for (u32 i = 0; i < count; ++i) {
		out[i].f = { raw[i*8+0], raw[i*8+1], raw[i*8+2] };
		out[i].pdf = raw[i*8+3];
		out[i].reference = { raw[i*8+4], raw[i*8+5], raw[i*8+6] };
	}
	return out;
}

// -----------------------------------------------------------------------------
// thin_walled_subsurface_integrate.comp runner
// -----------------------------------------------------------------------------

struct IntegrateOut {
	f32v3 albedoReflect;
	f32v3 albedoTransmit;
	f32 pdfIntegral;
};

void push_integrate_config(
	std::vector<f32> & flat,
	f32v3 const wi, f32 const sigma, f32 const anisotropy,
	f32v3 const subsurfaceColor
) {
	flat.emplace_back(wi.x);
	flat.emplace_back(wi.y);
	flat.emplace_back(wi.z);
	flat.emplace_back(sigma);
	flat.emplace_back(anisotropy);
	flat.emplace_back(subsurfaceColor.x);
	flat.emplace_back(subsurfaceColor.y);
	flat.emplace_back(subsurfaceColor.z);
}

std::vector<IntegrateOut> run_integrate(
	std::vector<f32> const & configsFlat,
	u32 const thetaSteps,
	u32 const phiSteps
) {
	u32 const count = (u32)configsFlat.size() / 8u;
	REQUIRE(configsFlat.size() == (size_t)count * 8u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "thin_walled_subsurface_integrate.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto inBuf = vkof::buffer_create({
		.byteCount = configsFlat.size() * sizeof(f32),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = inBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(configsFlat.data()),
			configsFlat.size() * sizeof(f32)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = count * 8u * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push {
		u64 inVa;
		u64 outVa;
		u32 count;
		u32 thetaSteps;
		u32 phiSteps;
	};
	Push const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
		.thetaSteps = thetaSteps,
		.phiSteps = phiSteps,
	};
	test::dispatch(pl, push, groups_for(count, 16u));

	auto raw = test::readback<f32>(outBuf, 0, count * 8u);
	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);

	std::vector<IntegrateOut> out(count);
	for (u32 i = 0; i < count; ++i) {
		out[i].albedoReflect = { raw[i*8+0], raw[i*8+1], raw[i*8+2] };
		out[i].albedoTransmit = { raw[i*8+3], raw[i*8+4], raw[i*8+5] };
		out[i].pdfIntegral = raw[i*8+6];
	}
	return out;
}

// -----------------------------------------------------------------------------
// histogram / expected runners
// -----------------------------------------------------------------------------

struct SamplePush {
	u64 va;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 nor;
	f32 anisotropy;
};
static_assert(sizeof(SamplePush) == 32);

std::vector<u32> run_histogram(
	f32v3 const nor, f32 const anisotropy,
	u32 const sampleCount, u32 const seedSalt
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "thin_walled_subsurface_histogram.comp",
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

	SamplePush const push {
		.va = vkof::buffer_virtual_address(histBuf),
		.sampleCount = sampleCount,
		.seedSalt = seedSalt,
		.nor = nor,
		.anisotropy = anisotropy,
	};
	test::dispatch(pl, push, groups_for(sampleCount, 256u));

	auto out = test::readback<u32>(histBuf, 0, slotCount);
	vkof::buffer_destroy(histBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

std::vector<f32> run_expected(
	f32v3 const nor, f32 const anisotropy, u32 const sampleCount
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "thin_walled_subsurface_expected.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto buf = vkof::buffer_create({
		.byteCount = kBinCount * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	SamplePush const push {
		.va = vkof::buffer_virtual_address(buf),
		.sampleCount = sampleCount,
		.seedSalt = 0u,
		.nor = nor,
		.anisotropy = anisotropy,
	};
	test::dispatch(pl, push, groups_for(kBinCount, 256u));

	auto out = test::readback<f32>(buf, 0, kBinCount);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

// pearson chi-square + wilson-hilferty p-value, same convention as
// test-subsurface-entry-sampling.cpp
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

f32v3 direction_from_angles(f64 const theta, f64 const phi) {
	return {
		(f32)(std::sin(theta) * std::cos(phi)),
		(f32)(std::sin(theta) * std::sin(phi)),
		(f32)std::cos(theta),
	};
}

f32v3 mirror_through_plane(f32v3 const nor, f32v3 const v) {
	f32 const d = nor.x * v.x + nor.y * v.y + nor.z * v.z;
	return { v.x - 2.0f * d * nor.x, v.y - 2.0f * d * nor.y, v.z - 2.0f * d * nor.z };
}

} // namespace

TEST_SUITE("[headless]") {

// -----------------------------------------------------------------------------
// lobe shape and split
// -----------------------------------------------------------------------------

TEST_CASE("thin-walled subsurface: lobe shape matches the materialx EON reference") {
	std::vector<f32> flat;
	f32v3 const nor { 0.0f, 0.0f, 1.0f };
	std::vector<f32> const sigmas { 0.0f, 0.3f, 0.7f, 1.0f };
	std::vector<f32> const anisotropies { -0.8f, -0.3f, 0.0f, 0.4f, 0.9f };
	f32v3 const subsurfaceColor { 0.8f, 0.55f, 0.3f };

	std::vector<f32> expectedSplit;
	for (f32 const sigma : sigmas) {
		for (f32 const g : anisotropies) {
			for (u32 ti = 0; ti < 8u; ++ti) {
				f64 const thetaI = 0.05 + 1.45 * (f64)ti / 7.0;
				f32v3 const wi = direction_from_angles(thetaI, 0.4);
				for (u32 to = 0; to < 12u; ++to) {
					// spans both hemispheres
					f64 const thetaO = 0.05 + (kPi - 0.1) * (f64)to / 11.0;
					f32v3 const wo = direction_from_angles(thetaO, 2.1);
					push_evaluate_sample(flat, nor, wi, wo, sigma, g, subsurfaceColor);
					expectedSplit.emplace_back(
						0.5f * (wo.z < 0.0f ? 1.0f + g : 1.0f - g)
					);
				}
			}
		}
	}

	auto const out = run_evaluate(flat);
	REQUIRE(out.size() == expectedSplit.size());

	f64 worst = 0.0;
	for (size_t i = 0; i < out.size(); ++i) {
		f32 const split = expectedSplit[i];
		f32v3 const expectedF {
			subsurfaceColor.x * split * out[i].reference.x,
			subsurfaceColor.y * split * out[i].reference.y,
			subsurfaceColor.z * split * out[i].reference.z,
		};
		worst = std::fmax(worst, std::fabs((f64)out[i].f.x - expectedF.x));
		worst = std::fmax(worst, std::fabs((f64)out[i].f.y - expectedF.y));
		worst = std::fmax(worst, std::fabs((f64)out[i].f.z - expectedF.z));
		CHECK(out[i].f.x == doctest::Approx(expectedF.x).epsilon(0.002));
		CHECK(out[i].f.y == doctest::Approx(expectedF.y).epsilon(0.002));
		CHECK(out[i].f.z == doctest::Approx(expectedF.z).epsilon(0.002));
	}
	MESSAGE("materialx lobe-shape crosscheck max abs diff " << worst);
}

TEST_CASE("thin-walled subsurface: sheet looks identical from either side") {
	// the spec's thin-wall is the bulk structure mirrored about the base, so
	// flipping the normal and both directions must reproduce the same value
	std::vector<f32> flat;
	f32v3 const nor { 0.0f, 0.0f, 1.0f };
	f32v3 const norFlipped { 0.0f, 0.0f, -1.0f };
	f32v3 const subsurfaceColor { 0.7f, 0.7f, 0.7f };

	std::vector<f32v3> wis;
	std::vector<f32v3> wos;
	for (u32 ti = 0; ti < 6u; ++ti) {
		f64 const thetaI = 0.1 + 1.4 * (f64)ti / 5.0;
		for (u32 to = 0; to < 10u; ++to) {
			f64 const thetaO = 0.1 + (kPi - 0.2) * (f64)to / 9.0;
			wis.emplace_back(direction_from_angles(thetaI, 0.9));
			wos.emplace_back(direction_from_angles(thetaO, 3.3));
		}
	}
	for (size_t i = 0; i < wis.size(); ++i) {
		push_evaluate_sample(flat, nor, wis[i], wos[i], 0.5f, 0.35f, subsurfaceColor);
	}
	for (size_t i = 0; i < wis.size(); ++i) {
		push_evaluate_sample(
			flat, norFlipped,
			mirror_through_plane(nor, wis[i]), mirror_through_plane(nor, wos[i]),
			0.5f, 0.35f, subsurfaceColor
		);
	}

	auto const out = run_evaluate(flat);
	REQUIRE(out.size() == wis.size() * 2u);
	f64 worst = 0.0;
	for (size_t i = 0; i < wis.size(); ++i) {
		auto const & a = out[i].f;
		auto const & b = out[i + wis.size()].f;
		worst = std::fmax(worst, std::fabs((f64)a.x - b.x));
		CHECK(a.x == doctest::Approx(b.x).epsilon(0.0001));
		CHECK(a.y == doctest::Approx(b.y).epsilon(0.0001));
		CHECK(a.z == doctest::Approx(b.z).epsilon(0.0001));
	}
	MESSAGE("two-sidedness max abs diff " << worst);
}

// -----------------------------------------------------------------------------
// energy
// -----------------------------------------------------------------------------

TEST_CASE("thin-walled subsurface furnace: E_R + E_T == subsurface_color") {
	std::vector<f32> flat;
	std::vector<f32> const sigmas { 0.0f, 0.25f, 0.6f, 1.0f };
	std::vector<f32> const anisotropies { -0.9f, -0.5f, 0.0f, 0.5f, 0.9f };
	std::vector<f32> const cosines { 0.15f, 0.4f, 0.7f, 1.0f };
	f32v3 const subsurfaceColor { 0.9f, 0.6f, 0.35f };

	struct Config { f32 sigma; f32 g; f32 mu; };
	std::vector<Config> configs;
	for (f32 const sigma : sigmas) {
		for (f32 const g : anisotropies) {
			for (f32 const mu : cosines) {
				f64 const theta = std::acos((f64)mu);
				push_integrate_config(
					flat, direction_from_angles(theta, 0.0), sigma, g,
					subsurfaceColor
				);
				configs.emplace_back(Config { sigma, g, mu });
			}
		}
	}

	auto const out = run_integrate(flat, 256u, 512u);
	REQUIRE(out.size() == configs.size());

	f64 worstEnergy = 0.0;
	f64 worstSplit = 0.0;
	for (size_t i = 0; i < out.size(); ++i) {
		f32v3 const total {
			out[i].albedoReflect.x + out[i].albedoTransmit.x,
			out[i].albedoReflect.y + out[i].albedoTransmit.y,
			out[i].albedoReflect.z + out[i].albedoTransmit.z,
		};
		worstEnergy = std::fmax(
			worstEnergy, std::fabs((f64)total.x - subsurfaceColor.x)
		);
		INFO(
			"sigma=" << configs[i].sigma << " g=" << configs[i].g
			<< " mu=" << configs[i].mu
		);
		CHECK(total.x == doctest::Approx(subsurfaceColor.x).epsilon(0.01));
		CHECK(total.y == doctest::Approx(subsurfaceColor.y).epsilon(0.01));
		CHECK(total.z == doctest::Approx(subsurfaceColor.z).epsilon(0.01));

		// and the split itself must track g
		f64 const transmitFraction = (f64)out[i].albedoTransmit.x / total.x;
		f64 const expectedFraction = 0.5 * (1.0 + (f64)configs[i].g);
		worstSplit = std::fmax(
			worstSplit, std::fabs(transmitFraction - expectedFraction)
		);
		CHECK(transmitFraction == doctest::Approx(expectedFraction).epsilon(0.01));
	}
	MESSAGE("furnace worst |E_R + E_T - S| " << worstEnergy);
	MESSAGE("worst transmit-fraction deviation from (1+g)/2 " << worstSplit);
}

TEST_CASE("thin-walled subsurface: pdf integrates to 1 over the sphere") {
	std::vector<f32> flat;
	std::vector<f32> const anisotropies { -1.0f, -0.6f, 0.0f, 0.6f, 1.0f };
	for (f32 const g : anisotropies) {
		push_integrate_config(
			flat, f32v3 { 0.0f, 0.0f, 1.0f }, 0.4f, g, f32v3 { 1.0f, 1.0f, 1.0f }
		);
	}
	auto const out = run_integrate(flat, 256u, 512u);
	REQUIRE(out.size() == anisotropies.size());
	for (size_t i = 0; i < out.size(); ++i) {
		INFO("g=" << anisotropies[i]);
		CHECK(out[i].pdfIntegral == doctest::Approx(1.0f).epsilon(0.005));
	}
}

// -----------------------------------------------------------------------------
// sampler
// -----------------------------------------------------------------------------

TEST_CASE("thin-walled subsurface: sampler matches its own pdf (chi-square)") {
	constexpr u32 kSampleCount = 4000000u;
	std::vector<f32v3> const normals {
		{ 0.0f, 0.0f, 1.0f },
		{ 0.4082483f, 0.4082483f, 0.8164966f },
	};
	std::vector<f32> const anisotropies { -0.7f, 0.0f, 0.45f };

	for (f32v3 const & nor : normals) {
		for (f32 const g : anisotropies) {
			auto const observed = run_histogram(nor, g, kSampleCount, 7u);
			auto const expected = run_expected(nor, g, kSampleCount);
			INFO("nor=(" << nor.x << "," << nor.y << "," << nor.z << ") g=" << g);
			// non-finite / non-unit sampled directions land in the overflow slot
			CHECK(observed[kBinCount] == 0u);
			auto const cs = chi_square(observed, expected);
			CHECK((f64)cs.observedTotal == doctest::Approx(cs.expectedTotal).epsilon(0.001));
			CHECK(cs.pValue > 1e-4);
			MESSAGE(
				"g=" << g << " chi2=" << cs.statistic << " dof=" << cs.dof
				<< " p=" << cs.pValue
			);
		}
	}
}

TEST_CASE("thin-walled subsurface: g = +-1 collapses onto a single hemisphere") {
	constexpr u32 kSampleCount = 200000u;
	f32v3 const nor { 0.0f, 0.0f, 1.0f };

	// g = 1: pure transmission, every sample below the horizon
	auto const transmitOnly = run_histogram(nor, 1.0f, kSampleCount, 3u);
	u64 upper = 0;
	for (u32 bin = 0; bin < kBinCount; ++bin) {
		if (bin / kPhiBins < kThetaBins / 2u) { upper += transmitOnly[bin]; }
	}
	CHECK(upper == 0u);

	// g = -1: pure reflection, every sample above the horizon
	auto const reflectOnly = run_histogram(nor, -1.0f, kSampleCount, 3u);
	u64 lower = 0;
	for (u32 bin = 0; bin < kBinCount; ++bin) {
		if (bin / kPhiBins >= kThetaBins / 2u) { lower += reflectOnly[bin]; }
	}
	CHECK(lower == 0u);
}

// -----------------------------------------------------------------------------
// robustness + heatmaps
// -----------------------------------------------------------------------------

TEST_CASE("thin-walled subsurface: finite and non-negative over the domain (fuzz)") {
	std::vector<f32> flat;
	u32 seed = 0x51ed270bu;
	auto nextFloat = [&seed]() {
		seed = seed * 1664525u + 1013904223u;
		return (f32)((seed >> 8) & 0xffffffu) / (f32)0xffffff;
	};

	constexpr u32 kCount = 20000u;
	f32v3 const nor { 0.0f, 0.0f, 1.0f };
	for (u32 i = 0; i < kCount; ++i) {
		f64 const thetaI = (f64)nextFloat() * (kPi * 0.5 - 1e-4);
		f64 const thetaO = (f64)nextFloat() * kPi;
		f64 const phiI = (f64)nextFloat() * 2.0 * kPi;
		f64 const phiO = (f64)nextFloat() * 2.0 * kPi;
		push_evaluate_sample(
			flat, nor,
			direction_from_angles(thetaI, phiI),
			direction_from_angles(thetaO, phiO),
			nextFloat(),
			nextFloat() * 2.0f - 1.0f,
			f32v3 { nextFloat(), nextFloat(), nextFloat() }
		);
	}

	auto const out = run_evaluate(flat);
	u32 nonFinite = 0;
	u32 negative = 0;
	for (auto const & o : out) {
		bool const finite = (
			std::isfinite(o.f.x) && std::isfinite(o.f.y) && std::isfinite(o.f.z)
			&& std::isfinite(o.pdf)
		);
		if (!finite) { nonFinite++; }
		if (o.f.x < 0.0f || o.f.y < 0.0f || o.f.z < 0.0f || o.pdf < 0.0f) {
			negative++;
		}
	}
	CHECK(nonFinite == 0u);
	CHECK(negative == 0u);
}

TEST_CASE("thin-walled subsurface: eval heatmap (outgoing theta x anisotropy)") {
	constexpr u32 kWidth = 128u;
	constexpr u32 kHeight = 128u;
	std::vector<f32> flat;
	f32v3 const nor { 0.0f, 0.0f, 1.0f };
	f32v3 const wi = direction_from_angles(0.6, 0.0);
	f32v3 const subsurfaceColor { 0.9f, 0.6f, 0.35f };

	for (u32 y = 0; y < kHeight; ++y) {
		// anisotropy over [-1, 1]
		f32 const g = -1.0f + 2.0f * (f32)y / (f32)(kHeight - 1u);
		for (u32 x = 0; x < kWidth; ++x) {
			// outgoing theta over the full sphere, so the row crosses the
			// reflection/transmission boundary at the midpoint
			f64 const thetaO = kPi * (f64)x / (f64)(kWidth - 1u);
			push_evaluate_sample(
				flat, nor, wi, direction_from_angles(thetaO, 0.0),
				0.5f, g, subsurfaceColor
			);
		}
	}

	auto const out = run_evaluate(flat);
	REQUIRE(out.size() == kWidth * kHeight);
	std::vector<f32> r(kWidth * kHeight);
	std::vector<f32> gCh(kWidth * kHeight);
	std::vector<f32> b(kWidth * kHeight);
	for (u32 i = 0; i < kWidth * kHeight; ++i) {
		r[i] = out[i].f.x * (f32)kPi;
		gCh[i] = out[i].f.y * (f32)kPi;
		b[i] = out[i].f.z * (f32)kPi;
	}
	u32 nanCount = 0;
	REQUIRE(test::write_heatmap_png(
		r, gCh, b, kWidth, kHeight,
		SUBSURFACE_OUTPUT_DIR "thin_walled_subsurface_heatmap.png", &nanCount
	));
	CHECK(nanCount == 0u);
}

} // TEST_SUITE("[headless]")
