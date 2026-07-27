#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <cmath>
#include <vector>

namespace {

struct FresnelPush {
	u64 inVa;
	u64 outVa;
	u32 count;
};

static constexpr u32 kLocalSize = 64;
static u32 groups_for(u32 count) { return (count + kLocalSize - 1) / kLocalSize; }

std::vector<f32> evaluate_dielectric(std::vector<f32v2> const & samples) {
	u32 const count = (u32)samples.size();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fresnel_dielectric_evaluate.comp",
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
		.byteCount = count * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	FresnelPush const push {
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

// each sample is 8 floats: f0.xyz, specularWeight, specularColor.xyz, dotNorWi
std::vector<f32v3> evaluate_metallic(std::vector<f32> const & samplesFlat) {
	u32 const count = (u32)samplesFlat.size() / 8u;
	REQUIRE(samplesFlat.size() == (size_t)count * 8u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fresnel_metallic_evaluate.comp",
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

	FresnelPush const push {
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

// output is f32v2(mine, materialx) per sample
std::vector<f32v2> evaluate_dielectric_crosscheck(
	std::vector<f32v2> const & samples
) {
	u32 const count = (u32)samples.size();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fresnel_dielectric_crosscheck.comp",
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

	FresnelPush const push {
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

struct MetallicCrosscheckPush {
	u64 inVa;
	u64 mineOutVa;
	u64 refOutVa;
	u32 count;
};

struct MetallicCrosscheck { std::vector<f32v3> mine, ref; };

MetallicCrosscheck evaluate_metallic_crosscheck(std::vector<f32> const & samplesFlat) {
	u32 const count = (u32)samplesFlat.size() / 8u;
	REQUIRE(samplesFlat.size() == (size_t)count * 8u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fresnel_metallic_crosscheck.comp",
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
	auto mineOutBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v3),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	auto refOutBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v3),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	MetallicCrosscheckPush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.mineOutVa = vkof::buffer_virtual_address(mineOutBuf),
		.refOutVa = vkof::buffer_virtual_address(refOutBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	MetallicCrosscheck out;
	out.mine = test::readback<f32v3>(mineOutBuf, 0, count);
	out.ref = test::readback<f32v3>(refOutBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(mineOutBuf);
	vkof::buffer_destroy(refOutBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

void push_metallic_sample(
	std::vector<f32> & flat,
	f32v3 f0, f32 specularWeight, f32v3 specularColor, f32 dotNorWi
) {
	flat.push_back(f0.x);
	flat.push_back(f0.y);
	flat.push_back(f0.z);
	flat.push_back(specularWeight);
	flat.push_back(specularColor.x);
	flat.push_back(specularColor.y);
	flat.push_back(specularColor.z);
	flat.push_back(dotNorWi);
}

void write_metallic_heatmap(f32v3 f0, f32v3 specularColor, char const * path) {
	// x: dotNorWi, 0 -> 1. y: specularWeight, 0 -> 1
	constexpr u32 kWidth = 512;
	constexpr u32 kHeight = 256;

	std::vector<f32> flat;
	flat.reserve((size_t)kWidth * kHeight * 8);
	for (u32 y = 0; y < kHeight; ++y) {
		f32 const specularWeight = (f32)y / (f32)(kHeight - 1);
		for (u32 x = 0; x < kWidth; ++x) {
			f32 const dotNorWi = (f32)x / (f32)(kWidth - 1);
			push_metallic_sample(flat, f0, specularWeight, specularColor, dotNorWi);
		}
	}
	auto const result = evaluate_metallic(flat);

	std::vector<f32> r(result.size()), g(result.size()), b(result.size());
	for (u32 i = 0; i < result.size(); ++i) {
		r[i] = result[i].x; g[i] = result[i].y; b[i] = result[i].z;
	}
	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(r, g, b, kWidth, kHeight, path, &nanCount);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);
}

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("fresnel: dielectric golden values at normal/grazing/matched incidence") {
	struct Case { f32 eta; };
	std::vector<Case> const cases = {
		{1.5f}, {1.0f / 1.5f}, {1.33f}, {2.42f}, {1.0f},
	};

	std::vector<f32v2> samples;
	for (auto const & c : cases) {
		samples.push_back({ 1.0f, c.eta });   // normal incidence
		samples.push_back({ 1e-4f, c.eta });  // near-grazing
	}
	auto const result = evaluate_dielectric(samples);

	for (u32 i = 0; i < cases.size(); ++i) {
		f32 const eta = cases[i].eta;
		f32 const expectedNormal = (
			((eta - 1.0f) / (eta + 1.0f)) * ((eta - 1.0f) / (eta + 1.0f))
		);
		CAPTURE(eta);
		CHECK(result[i * 2 + 0] == doctest::Approx(expectedNormal).epsilon(0.001));
		CHECK(result[i * 2 + 1] == doctest::Approx(1.0).epsilon(0.01));
	}
}

TEST_CASE("fresnel: dielectric is zero at all angles when ior is matched") {
	std::vector<f32v2> samples;
	for (f32 ct = 0.05f; ct <= 1.0f; ct += 0.05f) { samples.push_back({ ct, 1.0f }); }
	auto const result = evaluate_dielectric(samples);

	for (f32 v : result) {
		CAPTURE(v);
		CHECK(v == doctest::Approx(0.0).epsilon(0.001));
	}
}

TEST_CASE("fresnel: dielectric total internal reflection has no nans and clamps to 1") {
	f32 const eta = 1.0f / 1.5f; // glass -> air, eta < 1
	f32 const cosThetaCritical = std::sqrt(1.0f - eta * eta);

	std::vector<f32v2> samples;
	std::vector<f32> cosThetas;
	for (f32 ct = 0.001f; ct <= 1.0f; ct += 0.001f) {
		samples.push_back({ ct, eta });
		cosThetas.push_back(ct);
	}
	auto const result = evaluate_dielectric(samples);

	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(cosThetas[i]);
		CHECK(std::isfinite(result[i]));
		if (cosThetas[i] < cosThetaCritical) {
			CHECK(result[i] == doctest::Approx(1.0).epsilon(0.0001));
		}
	}
}

TEST_CASE("fresnel: dielectric stays in [0,1] and is monotonic in theta (fuzz)") {
	std::vector<f32> const etas = { 0.3f, 0.5f, 0.8f, 1.0f, 1.33f, 1.5f, 2.42f, 3.0f };

	for (f32 eta : etas) {
		std::vector<f32v2> samples;
		for (f32 ct = 1.0f; ct >= 0.0f; ct -= 0.02f) { samples.push_back({ ct, eta }); }
		auto const result = evaluate_dielectric(samples);

		f32 prev = -1.0f;
		for (u32 i = 0; i < result.size(); ++i) {
			CAPTURE(eta);
			CAPTURE(i);
			CHECK(result[i] >= 0.0f);
			CHECK(result[i] <= 1.0001f);
			// monotonic non-decreasing as theta increases (cosTheta decreases)
			CHECK(result[i] >= prev - 0.001f);
			prev = result[i];
		}
	}
}

TEST_CASE("fresnel: dielectric is reciprocal across the interface") {
	std::vector<f32> const etas = { 1.2f, 1.33f, 1.5f, 2.0f, 2.42f };
	std::vector<f32> const cosThetaIs = { 0.2f, 0.4f, 0.6f, 0.8f, 0.95f };

	std::vector<f32v2> forward, backward;
	for (f32 eta : etas) {
		for (f32 cosThetaI : cosThetaIs) {
			f32 const sinThetaT2 = (1.0f - cosThetaI * cosThetaI) / (eta * eta);
			if (sinThetaT2 > 1.0f) { continue; } // skip TIR, not applicable here
			f32 const cosThetaT = std::sqrt(1.0f - sinThetaT2);
			forward.push_back({ cosThetaI, eta });
			backward.push_back({ cosThetaT, 1.0f / eta });
		}
	}
	auto const fwd = evaluate_dielectric(forward);
	auto const bwd = evaluate_dielectric(backward);

	for (u32 i = 0; i < fwd.size(); ++i) {
		CAPTURE(i);
		CHECK(fwd[i] == doctest::Approx(bwd[i]).epsilon(0.001));
	}
}

TEST_CASE("fresnel: f82-tint conductor matches golden values at anchor points") {
	f32v3 const f0 { 0.7f, 0.4f, 0.2f };
	f32 const specularWeight = 0.6f;
	f32v3 const specularColor { 0.9f, 0.8f, 0.7f };

	std::vector<f32> flat;
	push_metallic_sample(flat, f0, specularWeight, specularColor, 1.0f);      // mu=1
	push_metallic_sample(flat, f0, specularWeight, specularColor, 0.0f);      // mu=0
	push_metallic_sample(flat, f0, specularWeight, specularColor, 1.0f/7.0f); // mu=1/7
	auto const result = evaluate_metallic(flat);

	// mu=1: F == f0 exactly
	CHECK(result[0].x == doctest::Approx(f0.x).epsilon(0.0001));
	CHECK(result[0].y == doctest::Approx(f0.y).epsilon(0.0001));
	CHECK(result[0].z == doctest::Approx(f0.z).epsilon(0.0001));

	// mu=0: F == white
	CHECK(result[1].x == doctest::Approx(1.0).epsilon(0.0001));
	CHECK(result[1].y == doctest::Approx(1.0).epsilon(0.0001));
	CHECK(result[1].z == doctest::Approx(1.0).epsilon(0.0001));

	// mu=1/7: F == fs(1/7) * specularWeight * specularColor, where
	// fs(1/7) = f0 + (1-f0)*(6/7)^5 -- derived from the formula's own
	// construction (fresnelConst term is designed to fully cancel fs(1/7)
	// except for this factor)
	f32 const schlickAt1_7 = std::pow(6.0f / 7.0f, 5.0f);
	f32v3 const fs17 {
		f0.x + (1.0f - f0.x) * schlickAt1_7,
		f0.y + (1.0f - f0.y) * schlickAt1_7,
		f0.z + (1.0f - f0.z) * schlickAt1_7,
	};
	CHECK(
		result[2].x
		== doctest::Approx(fs17.x * specularWeight * specularColor.x).epsilon(0.001)
	);
	CHECK(
		result[2].y
		== doctest::Approx(fs17.y * specularWeight * specularColor.y).epsilon(0.001)
	);
	CHECK(
		result[2].z
		== doctest::Approx(fs17.z * specularWeight * specularColor.z).epsilon(0.001)
	);
}

TEST_CASE("fresnel: f82-tint reduces to plain schlick at full specular weight/color") {
	f32v3 const f0 { 0.5f, 0.3f, 0.9f };

	std::vector<f32> flat;
	std::vector<f32> mus;
	for (f32 mu = 0.0f; mu <= 1.0f; mu += 0.05f) {
		push_metallic_sample(flat, f0, 1.0f, { 1.0f, 1.0f, 1.0f }, mu);
		mus.push_back(mu);
	}
	auto const result = evaluate_metallic(flat);

	for (u32 i = 0; i < result.size(); ++i) {
		f32 const mu = mus[i];
		f32 const schlick = std::pow(1.0f - mu, 5.0f);
		f32v3 const expected {
			f0.x + (1.0f - f0.x) * schlick,
			f0.y + (1.0f - f0.y) * schlick,
			f0.z + (1.0f - f0.z) * schlick,
		};
		CAPTURE(mu);
		CHECK(result[i].x == doctest::Approx(expected.x).epsilon(0.001));
		CHECK(result[i].y == doctest::Approx(expected.y).epsilon(0.001));
		CHECK(result[i].z == doctest::Approx(expected.z).epsilon(0.001));
	}
}

TEST_CASE("fresnel: f82-tint conductor has no nans for plausible inputs (fuzz)") {
	std::vector<f32> flat;
	u32 seed = 12345u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	constexpr u32 kCount = 512;
	for (u32 i = 0; i < kCount; ++i) {
		push_metallic_sample(
			flat,
			{ next(), next(), next() },
			next(),
			{ next(), next(), next() },
			next()
		);
	}
	auto const result = evaluate_metallic(flat);

	for (auto const & v : result) {
		CHECK(std::isfinite(v.x));
		CHECK(std::isfinite(v.y));
		CHECK(std::isfinite(v.z));
	}
}

TEST_CASE("fresnel: dielectric heatmap (angle x ior), nans marked red") {
	// x: cosTheta, 1 (normal incidence) -> 0 (grazing)
	// y: eta, spans both eta<1 (tir visible) and eta>1; eta=1 row should
	// be solid black (zero reflectance at every angle)
	constexpr u32 kWidth = 512;
	constexpr u32 kHeight = 512;
	constexpr f32 kEtaMin = 0.4f;
	constexpr f32 kEtaMax = 2.5f;

	std::vector<f32v2> samples(kWidth * kHeight);
	for (u32 y = 0; y < kHeight; ++y) {
		f32 const eta = kEtaMin + (kEtaMax - kEtaMin) * (f32)y / (f32)(kHeight - 1);
		for (u32 x = 0; x < kWidth; ++x) {
			f32 const cosTheta = 1.0f - (f32)x / (f32)(kWidth - 1);
			samples[y * kWidth + x] = { cosTheta, eta };
		}
	}
	auto const result = evaluate_dielectric(samples);

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		result, result, result, kWidth, kHeight,
		FRESNEL_OUTPUT_DIR "fresnel_dielectric_heatmap.png", &nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);
}

TEST_CASE("fresnel: metallic heatmap, gold-like tint (angle x specular weight)") {
	write_metallic_heatmap(
		{ 1.00f, 0.86f, 0.57f }, { 1.00f, 0.65f, 0.25f },
		FRESNEL_OUTPUT_DIR "fresnel_metallic_heatmap_gold_tinted.png"
	);
}

TEST_CASE("fresnel: metallic heatmap, white/untinted (angle x specular weight)") {
	write_metallic_heatmap(
		{ 1.00f, 0.86f, 0.57f }, { 1.0f, 1.0f, 1.0f },
		FRESNEL_OUTPUT_DIR "fresnel_metallic_heatmap_untinted.png"
	);
}

TEST_CASE("fresnel: dielectric matches materialx's independent derivation") {
	// materialx's mx_fresnel_dielectric uses seb lagarde's "g" formulation,
	// a different algebraic path than utilFresnelDielectric's snell's-law
	// form. skips exact grazing (cosTheta=0) to avoid the shared, genuine
	// 0/0 singularity at eta=1 both derivations have there.
	std::vector<f32> const etas = { 0.3f, 0.5f, 0.8f, 1.0f, 1.33f, 1.5f, 2.42f, 3.0f };

	for (f32 eta : etas) {
		std::vector<f32v2> samples;
		for (f32 ct = 1.0f; ct >= 0.01f; ct -= 0.01f) { samples.push_back({ ct, eta }); }
		auto const result = evaluate_dielectric_crosscheck(samples);

		for (auto const & pair : result) {
			CAPTURE(eta);
			CHECK(pair.x == doctest::Approx(pair.y).epsilon(0.001));
		}
	}
}

TEST_CASE("fresnel: f82-tint matches materialx's hoffman-schlick generalization") {
	// materialx's mx_fresnel_hoffman_schlick generalizes f82-tint with
	// arbitrary F90/exponent; specializing to F90=white, exponent=5 (which
	// is what this call does) should reduce to exactly openpbr's formula,
	// with materialx's F82 param equal to specularWeight*specularColor
	std::vector<f32v3> const f0s = {
		{ 1.00f, 0.86f, 0.57f }, { 0.95f, 0.64f, 0.54f }, { 0.2f, 0.2f, 0.2f },
	};
	std::vector<f32v3> const tints = {
		{ 1.00f, 0.65f, 0.25f }, { 1.0f, 1.0f, 1.0f }, { 0.5f, 0.7f, 0.9f },
	};

	std::vector<f32> flat;
	for (auto const & f0 : f0s) {
		for (auto const & tint : tints) {
			for (f32 sw = 0.0f; sw <= 1.0f; sw += 0.1f) {
				for (f32 mu = 0.0f; mu <= 1.0f; mu += 0.02f) {
					push_metallic_sample(flat, f0, sw, tint, mu);
				}
			}
		}
	}
	auto const result = evaluate_metallic_crosscheck(flat);

	for (u32 i = 0; i < result.mine.size(); ++i) {
		CAPTURE(i);
		CHECK(result.mine[i].x == doctest::Approx(result.ref[i].x).epsilon(0.001));
		CHECK(result.mine[i].y == doctest::Approx(result.ref[i].y).epsilon(0.001));
		CHECK(result.mine[i].z == doctest::Approx(result.ref[i].z).epsilon(0.001));
	}
}

} // TEST_SUITE("[headless]")
