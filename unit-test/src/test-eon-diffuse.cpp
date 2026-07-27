#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

// tests for the energy-preserving oren-nayer (EON) glossy-diffuse lobe,
// openPbrGlossyDiffuseOrenNayerEvaluateF + utilOrenNayerEnergyCompensate
// (portsmouth 2024, openpbr glossy-diffuse slab).
//
// cross-checked against materialx's mx_oren_nayar_compensated_diffuse,
// which implements the same EON model but computes the fujii directional
// albedo E_FON with the exact closed form instead of ponder's portsmouth
// polynomial fit -- so the crosscheck also bounds the fit error.

namespace {

static constexpr u32 kLocalSize = 64;
static u32 groups_for(u32 count) { return (count + kLocalSize - 1) / kLocalSize; }

constexpr f64 kPi = 3.14159265358979323846;
constexpr f64 kFon1 = 0.5 - 2.0 / (3.0 * kPi);
constexpr f64 kFon2 = 2.0 / 3.0 - 28.0 / (15.0 * kPi);

// -----------------------------------------------------------------------------
// cpu f64 reference, transcribed from the same EON formulas the shader
// ports (openpbr spec / portsmouth 2024). not independent of the
// implementation -- it validates the gpu f32 evaluation against the
// intended math at double precision; independence comes from the
// materialx crosscheck below
// -----------------------------------------------------------------------------

f64 ref_energy_comp(f64 mu, f64 r) {
	f64 const mucomp = 1.0 - mu;
	f64 const g1 = 0.0571085289;
	f64 const g2 = 0.491881867;
	f64 const g3 = -0.332181442;
	f64 const g4 = 0.0714429953;
	f64 const gOverPi = (
		mucomp * (g1 + mucomp * (g2 + mucomp * (g3 + mucomp * g4)))
	);
	return (1.0 + r * gOverPi) / (1.0 + r * kFon1);
}

// one color channel of f_diffuse = f_EON + f^Fujii_EON
f64 ref_eon_f(f64 muI, f64 muO, f64 dotWiWo, f64 sigma, f64 w, f64 c) {
	f64 const s = dotWiWo - muI * muO;
	f64 const sOverT = (
		s > 0.0 ? s / std::max(std::max(muI, muO), (f64)1e-7) : s
	);
	f64 const A = 1.0 / (1.0 + kFon1 * sigma);
	f64 const single = c * w / kPi * A * (1.0 + sigma * sOverT);
	f64 const eWi = ref_energy_comp(muI, sigma);
	f64 const eWo = ref_energy_comp(muO, sigma);
	f64 const avg = A * (1.0 + kFon2 * sigma);
	f64 const rhoMs = (
		c * c * (avg / std::max(1.0 - avg, (f64)1e-6))
		/ (1.0 - c * (1.0 - avg))
	);
	f64 const multi = (
		w * rhoMs / kPi
		* std::max((f64)1e-7, 1.0 - eWi)
		* std::max((f64)1e-7, 1.0 - eWo)
	);
	return single + multi;
}

// -----------------------------------------------------------------------------
// gpu dispatch helpers
// -----------------------------------------------------------------------------

void push_eon_sample(
	std::vector<f32> & flat,
	f32v3 wi, f32v3 wo, f32 sigma, f32 baseWeight, f32v3 baseColor
) {
	flat.push_back(wi.x); flat.push_back(wi.y); flat.push_back(wi.z);
	flat.push_back(wo.x); flat.push_back(wo.y); flat.push_back(wo.z);
	flat.push_back(sigma);
	flat.push_back(baseWeight);
	flat.push_back(baseColor.x);
	flat.push_back(baseColor.y);
	flat.push_back(baseColor.z);
}

std::vector<f32v3> evaluate_eon(std::vector<f32> const & samplesFlat) {
	u32 const count = (u32)samplesFlat.size() / 11u;
	REQUIRE(samplesFlat.size() == (size_t)count * 11u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "eon_diffuse_evaluate.comp",
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
		.byteCount = count * sizeof(f32v3),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push { u64 inVa; u64 outVa; u32 count; };
	Push const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	auto out = test::readback<f32v3>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

// mine = ponder's f; ref = materialx's raw brdf (no 1/pi, no weight):
// expected relation mine == baseWeight * (1/pi) * ref
void crosscheck_eon(
	std::vector<f32> const & samplesFlat,
	std::vector<f32v3> & outMine,
	std::vector<f32v3> & outRef
) {
	u32 const count = (u32)samplesFlat.size() / 11u;
	REQUIRE(samplesFlat.size() == (size_t)count * 11u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "eon_diffuse_crosscheck.comp",
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
	auto mineBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v3),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	auto refBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v3),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push { u64 inVa; u64 mineOutVa; u64 refOutVa; u32 count; };
	Push const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.mineOutVa = vkof::buffer_virtual_address(mineBuf),
		.refOutVa = vkof::buffer_virtual_address(refBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	outMine = test::readback<f32v3>(mineBuf, 0, count);
	outRef = test::readback<f32v3>(refBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(mineBuf);
	vkof::buffer_destroy(refBuf);
	vkof::pipeline_destroy(pl);
}

// per sample f32v2(mu, sigma) -> f32v2(portsmouth poly, materialx exact)
std::vector<f32v2> crosscheck_energy(std::vector<f32v2> const & samples) {
	u32 const count = (u32)samples.size();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "eon_energy_comp_crosscheck.comp",
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
			reinterpret_cast<u8 const *>(samples.data()),
			count * sizeof(f32v2)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v2),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push { u64 inVa; u64 outVa; u32 count; };
	Push const push {
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

// per sample f32v3(muI, sigma, colorScalar) -> hemispherical albedo
std::vector<f32> integrate_furnace(
	std::vector<f32v3> const & samples, u32 thetaSteps, u32 phiSteps
) {
	u32 const count = (u32)samples.size();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "eon_furnace_integrate.comp",
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

	struct Push { u64 inVa; u64 outVa; u32 thetaSteps; u32 phiSteps; u32 count; };
	Push const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.thetaSteps = thetaSteps,
		.phiSteps = phiSteps,
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	auto out = test::readback<f32>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

f32v3 dir_from(f32 cosTheta, f32 phi) {
	f32 const sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
	return { sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta };
}

} // namespace

TEST_SUITE("[headless]") {

// -----------------------------------------------------------------------------
// golden values
// -----------------------------------------------------------------------------

TEST_CASE("eon diffuse: sigma=0 reduces to lambert") {
	std::vector<f32> const weights = { 0.5f, 1.0f };
	std::vector<f32> const grays = { 0.18f, 0.5f, 1.0f };
	std::vector<f32> const cosines = { 0.05f, 0.3f, 0.7f, 1.0f };

	std::vector<f32> flat;
	std::vector<f64> expected;
	for (f32 w : weights) {
		for (f32 c : grays) {
			for (f32 muI : cosines) {
				for (f32 muO : cosines) {
					push_eon_sample(
						flat, dir_from(muI, 0.0f), dir_from(muO, 2.1f),
						0.0f, w, { c, c, c }
					);
					expected.push_back((f64)w * (f64)c / kPi);
				}
			}
		}
	}
	auto const result = evaluate_eon(flat);

	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(result[i].x == doctest::Approx(expected[i]).epsilon(0.001));
		CHECK(result[i].y == doctest::Approx(expected[i]).epsilon(0.001));
		CHECK(result[i].z == doctest::Approx(expected[i]).epsilon(0.001));
	}
}

TEST_CASE("eon diffuse: golden values at normal incidence (wi == wo == n)") {
	// s = dot(wi,wo) - muI muO = 0 exactly, so the single-scatter lobe is
	// w C A / pi with no azimuthal term; multiscatter adds on top
	std::vector<f32> const sigmas = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };
	std::vector<f32> const grays = { 0.2f, 0.6f, 1.0f };

	std::vector<f32> flat;
	std::vector<f64> expected;
	for (f32 sigma : sigmas) {
		for (f32 c : grays) {
			push_eon_sample(
				flat, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f },
				sigma, 1.0f, { c, c, c }
			);
			expected.push_back(ref_eon_f(1.0, 1.0, 1.0, sigma, 1.0, c));
		}
	}
	auto const result = evaluate_eon(flat);

	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(result[i].x == doctest::Approx(expected[i]).epsilon(0.001));
	}
}

TEST_CASE("eon diffuse: matches cpu f64 reference over a dense grid") {
	std::vector<f32> const sigmas = { 0.0f, 0.3f, 0.6f, 1.0f };
	std::vector<f32> const cosines = { 0.02f, 0.2f, 0.5f, 0.8f, 1.0f };
	std::vector<f32> const phis = { 0.0f, 1.0f, 2.0f, 3.14159265f };
	f32v3 const color = { 0.9f, 0.5f, 0.1f };

	std::vector<f32> flat;
	std::vector<f32v3> wis, wos;
	std::vector<f32> sigmasOut;
	for (f32 sigma : sigmas) {
		for (f32 muI : cosines) {
			for (f32 muO : cosines) {
				for (f32 phi : phis) {
					f32v3 const wi = dir_from(muI, 0.0f);
					f32v3 const wo = dir_from(muO, phi);
					push_eon_sample(flat, wi, wo, sigma, 1.0f, color);
					wis.push_back(wi);
					wos.push_back(wo);
					sigmasOut.push_back(sigma);
				}
			}
		}
	}
	auto const result = evaluate_eon(flat);

	for (u32 i = 0; i < result.size(); ++i) {
		f64 const dotWiWo = (
			(f64)wis[i].x * wos[i].x + (f64)wis[i].y * wos[i].y
			+ (f64)wis[i].z * wos[i].z
		);
		f64 const eX = ref_eon_f(
			wis[i].z, wos[i].z, dotWiWo, sigmasOut[i], 1.0, color.x
		);
		f64 const eZ = ref_eon_f(
			wis[i].z, wos[i].z, dotWiWo, sigmasOut[i], 1.0, color.z
		);
		CAPTURE(i);
		CHECK(result[i].x == doctest::Approx(eX).epsilon(0.002));
		CHECK(result[i].z == doctest::Approx(eZ).epsilon(0.002));
	}
}

// -----------------------------------------------------------------------------
// physical properties
// -----------------------------------------------------------------------------

TEST_CASE("eon diffuse: reciprocity f(wi,wo) == f(wo,wi) (fuzz)") {
	std::vector<f32> forward, swapped;
	u32 seed = 17u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 1024u; ++i) {
		f32v3 const wi = dir_from(0.01f + next() * 0.99f, next() * 6.2831853f);
		f32v3 const wo = dir_from(0.01f + next() * 0.99f, next() * 6.2831853f);
		f32 const sigma = next();
		f32v3 const c = { next(), next(), next() };
		push_eon_sample(forward, wi, wo, sigma, 1.0f, c);
		push_eon_sample(swapped, wo, wi, sigma, 1.0f, c);
	}
	auto const fwd = evaluate_eon(forward);
	auto const swp = evaluate_eon(swapped);

	for (u32 i = 0; i < fwd.size(); ++i) {
		CAPTURE(i);
		CHECK(fwd[i].x == doctest::Approx(swp[i].x).epsilon(0.0001));
		CHECK(fwd[i].y == doctest::Approx(swp[i].y).epsilon(0.0001));
		CHECK(fwd[i].z == doctest::Approx(swp[i].z).epsilon(0.0001));
	}
}

TEST_CASE("eon diffuse: non-negative and finite, grazing included (fuzz)") {
	std::vector<f32> flat;
	u32 seed = 4242u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 2048u; ++i) {
		// bias a quarter of the samples hard against the horizon
		f32 const muI = (i % 4u == 0u) ? next() * 0.01f : next();
		f32 const muO = (i % 4u == 1u) ? next() * 0.01f : next();
		push_eon_sample(
			flat,
			dir_from(muI, next() * 6.2831853f),
			dir_from(muO, next() * 6.2831853f),
			next(), next(), { next(), next(), next() }
		);
	}
	// exact-grazing and exact-normal corners
	push_eon_sample(
		flat, dir_from(0.0f, 0.0f), dir_from(0.0f, 0.0f), 1.0f, 1.0f,
		{ 1.0f, 1.0f, 1.0f }
	);
	push_eon_sample(
		flat, dir_from(0.0f, 0.0f), dir_from(0.0f, 3.14159265f), 1.0f, 1.0f,
		{ 1.0f, 1.0f, 1.0f }
	);
	push_eon_sample(
		flat, { 0.0f, 0.0f, 1.0f }, dir_from(0.0f, 1.0f), 1.0f, 1.0f,
		{ 1.0f, 1.0f, 1.0f }
	);
	auto const result = evaluate_eon(flat);

	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(result[i].x >= 0.0f);
		CHECK(std::isfinite(result[i].x));
		CHECK(std::isfinite(result[i].y));
		CHECK(std::isfinite(result[i].z));
	}
}

TEST_CASE("eon diffuse: retroreflective, f non-increasing in azimuthal separation") {
	// oren-nayer retroreflects: at fixed polar angles, s (and with it f)
	// peaks at delta-phi = 0 (wo backscattered toward wi) and decays
	// monotonically toward delta-phi = pi
	std::vector<f32> const sigmas = { 0.2f, 0.5f, 0.8f, 1.0f };
	f32 const mu = 0.5f;
	constexpr u32 kPhiSteps = 64;

	for (f32 sigma : sigmas) {
		std::vector<f32> flat;
		for (u32 p = 0; p < kPhiSteps; ++p) {
			f32 const phi = 3.14159265f * (f32)p / (f32)(kPhiSteps - 1);
			push_eon_sample(
				flat, dir_from(mu, 0.0f), dir_from(mu, phi), sigma, 1.0f,
				{ 1.0f, 1.0f, 1.0f }
			);
		}
		auto const result = evaluate_eon(flat);

		f32 prev = 1e30f;
		for (u32 i = 0; i < result.size(); ++i) {
			CAPTURE(sigma);
			CAPTURE(i);
			CHECK(result[i].x <= prev * 1.0001f);
			prev = result[i].x;
		}
		// the retro peak must be a strict enhancement over the forward
		// direction for rough surfaces
		CHECK(result[0].x > result[kPhiSteps - 1].x);
	}
}

TEST_CASE("eon diffuse: white furnace, hemispherical albedo == 1") {
	// the defining property of the energy-preserving construction: with
	// baseColor = 1 the lobe integrates to exactly 1 at every incident
	// angle and roughness -- no darkening, no gain
	std::vector<f32> const mus = { 0.05f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f };
	std::vector<f32> const sigmas = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };

	std::vector<f32v3> samples;
	for (f32 mu : mus) {
		for (f32 sigma : sigmas) {
			samples.push_back({ mu, sigma, 1.0f });
		}
	}
	auto const result = integrate_furnace(samples, 256u, 512u);

	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(samples[i].x);
		CAPTURE(samples[i].y);
		CHECK(result[i] == doctest::Approx(1.0).epsilon(0.01));
	}
}

TEST_CASE("eon diffuse: colored albedo bounded by 1 and monotonic in color") {
	std::vector<f32> const grays = { 0.2f, 0.5f, 0.8f, 1.0f };
	std::vector<f32> const mus = { 0.1f, 0.5f, 1.0f };
	f32 const sigma = 0.7f;

	std::vector<f32v3> samples;
	for (f32 mu : mus) {
		for (f32 c : grays) {
			samples.push_back({ mu, sigma, c });
		}
	}
	auto const result = integrate_furnace(samples, 256u, 512u);

	for (u32 m = 0; m < mus.size(); ++m) {
		f32 prev = -1.0f;
		for (u32 c = 0; c < grays.size(); ++c) {
			f32 const albedo = result[m * grays.size() + c];
			CAPTURE(mus[m]);
			CAPTURE(grays[c]);
			CHECK(albedo <= 1.001f);
			CHECK(albedo > prev);
			prev = albedo;
		}
	}
}

// -----------------------------------------------------------------------------
// energy compensation term (portsmouth polynomial fit)
// -----------------------------------------------------------------------------

TEST_CASE("eon energy comp: golden values") {
	// E(mu, 0) == 1 for any mu; E(1, r) == A(r) = 1/(1 + fon1 r) since the
	// polynomial's mucomp factor vanishes at normal incidence
	std::vector<f32v2> samples;
	std::vector<f64> expected;
	for (f32 mu : { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f }) {
		samples.push_back({ mu, 0.0f });
		expected.push_back(1.0);
	}
	for (f32 r : { 0.1f, 0.4f, 0.7f, 1.0f }) {
		samples.push_back({ 1.0f, r });
		expected.push_back(1.0 / (1.0 + kFon1 * (f64)r));
	}
	auto const result = crosscheck_energy(samples);

	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(result[i].x == doctest::Approx(expected[i]).epsilon(0.001));
	}
}

TEST_CASE("eon energy comp: bounded near (0, 1] and finite at exact grazing") {
	// the portsmouth fit overshoots 1 by up to ~4e-4 at exact grazing:
	// gOverPi(mucomp=1) = g1+g2+g3+g4 = 0.28824, a hair above
	// skFon1 = 0.28779, so E(0, r) = (1 + r*0.28824)/(1 + r*0.28779) > 1.
	// inherent to the published fit (cull and the openpbr reference share
	// it), and harmless downstream: the lobe clamps (1 - E) at 1e-7
	std::vector<f32v2> samples;
	for (u32 m = 0; m <= 100; ++m) {
		for (u32 r = 0; r <= 20; ++r) {
			samples.push_back({ (f32)m / 100.0f, (f32)r / 20.0f });
		}
	}
	auto const result = crosscheck_energy(samples);

	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(std::isfinite(result[i].x));
		CHECK(result[i].x > 0.0f);
		CHECK(result[i].x <= 1.001f);
	}
}

TEST_CASE("eon energy comp: polynomial fit matches materialx exact E_FON") {
	// materialx computes the fujii directional albedo with the exact
	// closed form; ponder uses the portsmouth polynomial fit. this bounds
	// the fit error. the exact form is 0/0 at mu == 0, so the grid stays
	// off exact grazing
	std::vector<f32v2> samples;
	for (u32 m = 0; m <= 200; ++m) {
		for (u32 r = 0; r <= 40; ++r) {
			samples.push_back({
				0.005f + 0.995f * (f32)m / 200.0f, (f32)r / 40.0f
			});
		}
	}
	auto const result = crosscheck_energy(samples);

	f64 maxAbsDiff = 0.0;
	u32 maxAt = 0;
	for (u32 i = 0; i < result.size(); ++i) {
		f64 const d = std::abs((f64)result[i].x - (f64)result[i].y);
		if (d > maxAbsDiff) { maxAbsDiff = d; maxAt = i; }
	}
	MESSAGE(
		"portsmouth-fit vs exact E_FON max abs diff ", maxAbsDiff,
		" at mu=", samples[maxAt].x, " sigma=", samples[maxAt].y
	);
	// the portsmouth 2024 fit is advertised as sub-percent; anything past
	// 0.015 absolute means the port (or the fit) is broken
	CHECK(maxAbsDiff < 0.015);
}

// -----------------------------------------------------------------------------
// materialx independent cross-check, full brdf
// -----------------------------------------------------------------------------

TEST_CASE("eon diffuse: matches materialx mx_oren_nayar_compensated_diffuse (fuzz)") {
	// mine == baseWeight * (1/pi) * materialx. the only modeling delta is
	// polynomial-vs-exact E_FON inside the multiscatter lobe, so the
	// tolerance is the fit error, not float noise
	std::vector<f32> flat;
	std::vector<f32> weights;
	u32 seed = 2024u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 2048u; ++i) {
		f32 const w = 0.25f + next() * 0.75f;
		push_eon_sample(
			flat,
			dir_from(0.02f + next() * 0.98f, next() * 6.2831853f),
			dir_from(0.02f + next() * 0.98f, next() * 6.2831853f),
			next(), w, { next(), next(), next() }
		);
		weights.push_back(w);
	}
	std::vector<f32v3> mine, ref;
	crosscheck_eon(flat, mine, ref);

	f64 maxAbsDiff = 0.0;
	u32 maxAt = 0;
	for (u32 i = 0; i < mine.size(); ++i) {
		f64 const scale = (f64)weights[i] / kPi;
		f64 const dx = std::abs((f64)mine[i].x - scale * (f64)ref[i].x);
		f64 const dy = std::abs((f64)mine[i].y - scale * (f64)ref[i].y);
		f64 const dz = std::abs((f64)mine[i].z - scale * (f64)ref[i].z);
		f64 const d = std::max(dx, std::max(dy, dz));
		if (d > maxAbsDiff) { maxAbsDiff = d; maxAt = i; }
	}
	MESSAGE("materialx fuzz crosscheck max abs diff ", maxAbsDiff);
	// brdf values here are O(1/pi); 0.005 absolute is ~1.5% of the lambert
	// level and comfortably above the polynomial fit residue
	CHECK(maxAbsDiff < 0.005);
}

TEST_CASE("eon diffuse: materialx crosscheck on a structured grid") {
	// deterministic complement to the fuzz: full sweep of polar angles at
	// a retro and a forward azimuth, white and mid-gray
	std::vector<f32> flat;
	std::vector<f32> const sigmas = { 0.0f, 0.5f, 1.0f };
	std::vector<f32> const grays = { 0.5f, 1.0f };
	for (f32 sigma : sigmas) {
		for (f32 gray : grays) {
			for (u32 a = 1; a <= 20; ++a) {
				for (u32 b = 1; b <= 20; ++b) {
					f32 const muI = (f32)a / 20.0f;
					f32 const muO = (f32)b / 20.0f;
					push_eon_sample(
						flat, dir_from(muI, 0.0f), dir_from(muO, 0.0f),
						sigma, 1.0f, { gray, gray, gray }
					);
					push_eon_sample(
						flat, dir_from(muI, 0.0f), dir_from(muO, 3.14159265f),
						sigma, 1.0f, { gray, gray, gray }
					);
				}
			}
		}
	}
	std::vector<f32v3> mine, ref;
	crosscheck_eon(flat, mine, ref);

	f64 maxAbsDiff = 0.0;
	for (u32 i = 0; i < mine.size(); ++i) {
		f64 const d = std::abs((f64)mine[i].x - (f64)ref[i].x / kPi);
		maxAbsDiff = std::max(maxAbsDiff, d);
	}
	MESSAGE("materialx grid crosscheck max abs diff ", maxAbsDiff);
	CHECK(maxAbsDiff < 0.005);
}

// -----------------------------------------------------------------------------
// visual representation
// -----------------------------------------------------------------------------

TEST_CASE("eon energy comp: heatmap (mu x sigma)") {
	// x: mu, 1 (normal) -> 0 (grazing). y: sigma, 0 -> 1. E is an energy
	// fraction in (0,1], so no tonemapping needed
	constexpr u32 kSize = 512;

	std::vector<f32v2> samples(kSize * kSize);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const sigma = (f32)y / (f32)(kSize - 1);
		for (u32 x = 0; x < kSize; ++x) {
			f32 const mu = 1.0f - (f32)x / (f32)(kSize - 1);
			samples[y * kSize + x] = { mu, sigma };
		}
	}
	auto const result = crosscheck_energy(samples);

	std::vector<f32> mine(result.size());
	for (u32 i = 0; i < result.size(); ++i) { mine[i] = result[i].x; }

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		mine, mine, mine, kSize, kSize,
		EON_OUTPUT_DIR "eon_energy_comp_heatmap.png", &nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);
}

TEST_CASE("eon energy comp: fit error heatmap vs exact E_FON (x100)") {
	// |portsmouth poly - exact| amplified 100x; a correct fit renders as
	// near-black. mu stays off 0 where the exact form is 0/0
	constexpr u32 kSize = 512;

	std::vector<f32v2> samples(kSize * kSize);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const sigma = (f32)y / (f32)(kSize - 1);
		for (u32 x = 0; x < kSize; ++x) {
			f32 const mu = 1.0f - 0.995f * (f32)x / (f32)(kSize - 1);
			samples[y * kSize + x] = { mu, sigma };
		}
	}
	auto const result = crosscheck_energy(samples);

	std::vector<f32> diff(result.size());
	for (u32 i = 0; i < result.size(); ++i) {
		diff[i] = std::abs(result[i].x - result[i].y) * 100.0f;
	}

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		diff, diff, diff, kSize, kSize,
		EON_OUTPUT_DIR "eon_energy_comp_fit_error_heatmap.png", &nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);
}

TEST_CASE("eon diffuse: white furnace heatmap (mu x sigma)") {
	// hemispherical albedo with baseColor = 1 across the whole parameter
	// square; energy preservation means a flat white image. rendered raw
	// (no tonemap): any darkening or gain is directly visible, non-finite
	// values would render red
	constexpr u32 kSize = 128;

	std::vector<f32v3> samples(kSize * kSize);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const sigma = (f32)y / (f32)(kSize - 1);
		for (u32 x = 0; x < kSize; ++x) {
			f32 const mu = 1.0f - 0.99f * (f32)x / (f32)(kSize - 1);
			samples[y * kSize + x] = { mu, sigma, 1.0f };
		}
	}
	auto const result = integrate_furnace(samples, 64u, 128u);

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		result, result, result, kSize, kSize,
		EON_OUTPUT_DIR "eon_furnace_heatmap.png", &nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);

	// the image doubles as a dense assertion sweep
	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(result[i] == doctest::Approx(1.0).epsilon(0.02));
	}
}

namespace {

void write_eon_disk(f32 sigma, char const * path) {
	// brdf * pi over the wo hemisphere projected to the tangent disk
	// (wo.x, wo.y), wi fixed at 60 degrees in the +x half-plane. the
	// retroreflection lobe shows as a bright region on the +x (wi) side.
	// outside the disk renders as background black
	constexpr u32 kSize = 512;
	f32 const muI = 0.5f;
	f32v3 const wi = { std::sqrt(1.0f - muI * muI), 0.0f, muI };

	std::vector<f32> flat;
	std::vector<bool> valid(kSize * kSize, false);
	flat.reserve((size_t)kSize * kSize * 11);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const wy = 2.0f * (f32)y / (f32)(kSize - 1) - 1.0f;
		for (u32 x = 0; x < kSize; ++x) {
			f32 const wx = 2.0f * (f32)x / (f32)(kSize - 1) - 1.0f;
			f32 const r2 = wx * wx + wy * wy;
			bool const inside = r2 <= 1.0f;
			valid[y * kSize + x] = inside;
			f32v3 const wo = {
				wx, wy, inside ? std::sqrt(1.0f - r2) : 1.0f
			};
			push_eon_sample(flat, wi, wo, sigma, 1.0f, { 1.0f, 1.0f, 1.0f });
		}
	}
	auto const result = evaluate_eon(flat);

	// f * pi is O(1); 1.5 headroom keeps the retro peak unclipped
	std::vector<f32> scaled(result.size(), 0.0f);
	for (u32 i = 0; i < result.size(); ++i) {
		if (valid[i]) {
			scaled[i] = result[i].x * 3.14159265f / 1.5f;
		}
	}

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		scaled, scaled, scaled, kSize, kSize, path, &nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);
}

} // namespace

TEST_CASE("eon diffuse: brdf disk heatmap, rough (retroreflection lobe)") {
	write_eon_disk(0.8f, EON_OUTPUT_DIR "eon_brdf_disk_rough.png");
}

TEST_CASE("eon diffuse: brdf disk heatmap, lambert control (sigma = 0)") {
	write_eon_disk(0.0f, EON_OUTPUT_DIR "eon_brdf_disk_lambert.png");
}

TEST_CASE("eon diffuse: materialx crosscheck diff disk heatmap (x50)") {
	// |mine - materialx/pi| * 50 over the same wo disk, sigma = 0.8; a
	// correct port renders near-black with only the polynomial fit residue
	constexpr u32 kSize = 512;
	f32 const sigma = 0.8f;
	f32 const muI = 0.5f;
	f32v3 const wi = { std::sqrt(1.0f - muI * muI), 0.0f, muI };

	std::vector<f32> flat;
	std::vector<bool> valid(kSize * kSize, false);
	flat.reserve((size_t)kSize * kSize * 11);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const wy = 2.0f * (f32)y / (f32)(kSize - 1) - 1.0f;
		for (u32 x = 0; x < kSize; ++x) {
			f32 const wx = 2.0f * (f32)x / (f32)(kSize - 1) - 1.0f;
			f32 const r2 = wx * wx + wy * wy;
			bool const inside = r2 <= 0.995f;
			valid[y * kSize + x] = inside;
			f32v3 const wo = {
				wx, wy, inside ? std::sqrt(1.0f - r2) : 1.0f
			};
			push_eon_sample(flat, wi, wo, sigma, 1.0f, { 1.0f, 1.0f, 1.0f });
		}
	}
	std::vector<f32v3> mine, ref;
	crosscheck_eon(flat, mine, ref);

	std::vector<f32> diff(mine.size(), 0.0f);
	for (u32 i = 0; i < mine.size(); ++i) {
		if (valid[i]) {
			diff[i] = (
				std::abs(mine[i].x - ref[i].x * (f32)(1.0 / kPi)) * 50.0f
			);
		}
	}

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		diff, diff, diff, kSize, kSize,
		EON_OUTPUT_DIR "eon_materialx_diff_disk.png", &nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);
}

} // TEST_SUITE("[headless]")
