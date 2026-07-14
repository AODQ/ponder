#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace {

struct LobePush {
	u64 inVa;
	u64 outVa;
	u32 count;
};

static constexpr u32 kLocalSize = 64;
static u32 groups_for(u32 count) { return (count + kLocalSize - 1) / kLocalSize; }

void push_lobe_sample(
	std::vector<f32> & flat,
	f32v3 wi, f32v3 wo,
	f32 specularRoughness, f32 anisotropy, f32 rotation,
	f32 coatRoughness, f32 coatWeight, f32 etaRel
) {
	flat.push_back(wi.x); flat.push_back(wi.y); flat.push_back(wi.z);
	flat.push_back(wo.x); flat.push_back(wo.y); flat.push_back(wo.z);
	flat.push_back(specularRoughness);
	flat.push_back(anisotropy);
	flat.push_back(rotation);
	flat.push_back(coatRoughness);
	flat.push_back(coatWeight);
	flat.push_back(etaRel);
}

std::vector<f32v3> evaluate_lobe(
	std::vector<f32> const & samplesFlat, char const * shaderName
) {
	u32 const count = (u32)samplesFlat.size() / 12u;
	REQUIRE(samplesFlat.size() == (size_t)count * 12u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	std::string const path = std::string(TEST_SHADER_DIR) + shaderName;
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = path.c_str(),
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

	LobePush const push {
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

std::vector<f32v3> evaluate_dielectric_specular(std::vector<f32> const & samplesFlat) {
	return evaluate_lobe(samplesFlat, "dielectric_specular_evaluate.comp");
}

std::vector<f32v3> evaluate_dielectric_specular_uncompensated(
	std::vector<f32> const & samplesFlat
) {
	return evaluate_lobe(samplesFlat, "dielectric_specular_uncompensated_evaluate.comp");
}

std::vector<f32> evaluate_furnace(
	std::vector<f32v3> const & samples, u32 thetaSteps, u32 phiSteps
) {
	u32 const count = (u32)samples.size();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "dielectric_specular_furnace_integrate.comp",
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

f32v3 random_hemisphere_vector(u32 & seed) {
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	f32 const z = 0.02f + next() * 0.96f; // keep away from exact grazing
	f32 const phi = next() * 6.28318530717958647692f;
	f32 const r = std::sqrt(std::max(0.0f, 1.0f - z * z));
	return { r * std::cos(phi), r * std::sin(phi), z };
}

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("dielectric specular: non-negative and finite (fuzz)") {
	std::vector<f32> flat;
	u32 seed = 1u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 2048u; ++i) {
		push_lobe_sample(
			flat, random_hemisphere_vector(seed), random_hemisphere_vector(seed),
			0.02f + next() * 0.98f, next(), next() * 6.28318530717958647692f,
			next(), next(), 0.5f + next() * 2.0f
		);
	}
	auto const result = evaluate_dielectric_specular(flat);
	for (auto const & f : result) {
		CHECK(std::isfinite(f.x));
		CHECK(std::isfinite(f.y));
		CHECK(std::isfinite(f.z));
		CHECK(f.x >= -0.0001f);
	}
}

TEST_CASE("dielectric specular: zero when eta=1 (matched ior)") {
	std::vector<f32> flat;
	u32 seed = 2u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 256u; ++i) {
		push_lobe_sample(
			flat, random_hemisphere_vector(seed), random_hemisphere_vector(seed),
			0.02f + next() * 0.98f, next(), next() * 6.28318530717958647692f,
			next(), next(), 1.0f
		);
	}
	auto const result = evaluate_dielectric_specular(flat);
	for (auto const & f : result) {
		CHECK(f.x == doctest::Approx(0.0).epsilon(0.0001));
	}
}

TEST_CASE("dielectric specular: uncompensated base lobe is reciprocal (fuzz)") {
	std::vector<f32> forward, swapped;
	u32 seed = 3u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 1024u; ++i) {
		f32v3 const wi = random_hemisphere_vector(seed);
		f32v3 const wo = random_hemisphere_vector(seed);
		f32 const roughness = 0.02f + next() * 0.98f;
		f32 const anisotropy = next();
		f32 const rotation = next() * 6.28318530717958647692f;
		f32 const etaRel = 0.5f + next() * 2.0f;
		push_lobe_sample(forward, wi, wo, roughness, anisotropy, rotation, 0.0f, 0.0f, etaRel);
		push_lobe_sample(swapped, wo, wi, roughness, anisotropy, rotation, 0.0f, 0.0f, etaRel);
	}
	auto const fwd = evaluate_dielectric_specular_uncompensated(forward);
	auto const swp = evaluate_dielectric_specular_uncompensated(swapped);

	for (u32 i = 0; i < fwd.size(); ++i) {
		CAPTURE(i);
		CHECK(fwd[i].x == doctest::Approx(swp[i].x).epsilon(0.001));
	}
}

TEST_CASE("dielectric specular: furnace test, reflected energy does not exceed 1") {
	constexpr u32 kThetaSteps = 300;
	constexpr u32 kPhiSteps = 150;
	std::vector<f32> const etaRels = { 1.0f / 1.5f, 1.33f, 1.5f, 2.42f };
	std::vector<f32> const roughnesses = { 0.1f, 0.3f, 0.5f, 0.8f };
	std::vector<f32> const mus = { 0.1f, 0.4f, 0.7f, 0.99f };

	std::vector<f32v3> samples;
	for (f32 etaRel : etaRels) {
		for (f32 roughness : roughnesses) {
			for (f32 mu : mus) { samples.push_back({ mu, roughness, etaRel }); }
		}
	}
	auto const result = evaluate_furnace(samples, kThetaSteps, kPhiSteps);

	f32 worst = 0.0f;
	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(samples[i].x);
		CAPTURE(samples[i].y);
		CAPTURE(samples[i].z);
		CAPTURE(result[i]);
		CHECK(std::isfinite(result[i]));
		CHECK(result[i] >= 0.0f);
		CHECK(result[i] <= 1.05f); // generous: turquin compensation is an approximation
		worst = std::max(worst, result[i]);
	}
	CAPTURE(worst);
}

TEST_CASE("dielectric specular: footprint (tangent-plane slice), eta=1.5") {
	// classic brdf-slice visualization: fix wi near-normal, sweep wo over
	// the tangent disc, render the (compensated) lobe response
	constexpr u32 kSize = 512;
	constexpr f32 kEtaRel = 1.5f;
	constexpr f32 kRoughness = 0.15f;
	f32v3 const wi = { 0.3f, 0.0f, std::sqrt(1.0f - 0.3f * 0.3f) };

	std::vector<f32> flat;
	std::vector<bool> valid(kSize * kSize, false);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const woy = 2.0f * (f32)y / (f32)(kSize - 1) - 1.0f;
		for (u32 x = 0; x < kSize; ++x) {
			f32 const wox = 2.0f * (f32)x / (f32)(kSize - 1) - 1.0f;
			f32 const r2 = wox * wox + woy * woy;
			bool const inside = r2 <= 1.0f;
			valid[y * kSize + x] = inside;
			f32 const woz = inside ? std::sqrt(1.0f - r2) : 1.0f;
			push_lobe_sample(
				flat, wi, { wox, woy, woz }, kRoughness, 0.0f, 0.0f, 0.0f, 0.0f, kEtaRel
			);
		}
	}
	auto const result = evaluate_dielectric_specular(flat);

	std::vector<f32> tonemapped(result.size(), 0.0f);
	for (u32 i = 0; i < result.size(); ++i) {
		if (valid[i]) { tonemapped[i] = result[i].x / (result[i].x + 1.0f); }
	}

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		tonemapped, tonemapped, tonemapped, kSize, kSize,
		DIELECTRIC_SPECULAR_OUTPUT_DIR "dielectric_specular_footprint.png", &nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);
}

} // TEST_SUITE("[headless]")
