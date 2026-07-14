#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <cmath>
#include <vector>

namespace {

struct GgxIsoPush {
	u64 inVa;
	u64 outVa;
	u32 count;
};

std::vector<f32> evaluate_ggx_iso(std::vector<f32v2> const & samples) {
	u32 const count = (u32)samples.size();
	static constexpr u32 kLocalSize = 64;
	u32 const groups = (count + kLocalSize - 1) / kLocalSize;

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "ggx_distribution_evaluate.comp",
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

	GgxIsoPush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups);

	auto out = test::readback<f32>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

// each sample is 5 floats: localH.xyz, alpha.xy
std::vector<f32> evaluate_ggx_aniso(std::vector<f32> const & samplesFlat) {
	u32 const count = (u32)samplesFlat.size() / 5u;
	REQUIRE(samplesFlat.size() == (size_t)count * 5u);
	static constexpr u32 kLocalSize = 64;
	u32 const groups = (count + kLocalSize - 1) / kLocalSize;

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "ggx_distribution_aniso_evaluate.comp",
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
		.byteCount = count * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	GgxIsoPush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups);

	auto out = test::readback<f32>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

// each sample is 5 floats: localH.xyz, alpha.xy. output f32v2(mine, materialx)
std::vector<f32v2> evaluate_ggx_aniso_crosscheck(std::vector<f32> const & samplesFlat) {
	u32 const count = (u32)samplesFlat.size() / 5u;
	REQUIRE(samplesFlat.size() == (size_t)count * 5u);
	static constexpr u32 kLocalSize = 64;
	u32 const groups = (count + kLocalSize - 1) / kLocalSize;

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "ggx_distribution_crosscheck.comp",
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
		.byteCount = count * sizeof(f32v2),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	GgxIsoPush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups);

	auto out = test::readback<f32v2>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

void push_aniso_sample(
	std::vector<f32> & flat, f32v3 localH, f32v2 alpha
) {
	flat.push_back(localH.x);
	flat.push_back(localH.y);
	flat.push_back(localH.z);
	flat.push_back(alpha.x);
	flat.push_back(alpha.y);
}

std::vector<f32> evaluate_smith_visibility(std::vector<f32v3> const & samples) {
	u32 const count = (u32)samples.size();
	static constexpr u32 kLocalSize = 64;
	u32 const groups = (count + kLocalSize - 1) / kLocalSize;

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "smith_visibility_evaluate.comp",
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
			reinterpret_cast<u8 const *>(samples.data()),
			count * sizeof(f32v3)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	GgxIsoPush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups);

	auto out = test::readback<f32>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

std::vector<f32> evaluate_smith_g1(std::vector<f32v2> const & samples) {
	u32 const count = (u32)samples.size();
	static constexpr u32 kLocalSize = 64;
	u32 const groups = (count + kLocalSize - 1) / kLocalSize;

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "smith_g1_evaluate.comp",
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

	GgxIsoPush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups);

	auto out = test::readback<f32>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

// each sample is 8 floats: wiLocal.xyz, woLocal.xyz, alpha.xy
std::vector<f32> evaluate_smith_visibility_aniso(std::vector<f32> const & samplesFlat) {
	u32 const count = (u32)samplesFlat.size() / 8u;
	REQUIRE(samplesFlat.size() == (size_t)count * 8u);
	static constexpr u32 kLocalSize = 64;
	u32 const groups = (count + kLocalSize - 1) / kLocalSize;

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "smith_visibility_aniso_evaluate.comp",
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
		.byteCount = count * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	GgxIsoPush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups);

	auto out = test::readback<f32>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

std::vector<f32v2> evaluate_smith_visibility_crosscheck(std::vector<f32v3> const & samples) {
	u32 const count = (u32)samples.size();
	static constexpr u32 kLocalSize = 64;
	u32 const groups = (count + kLocalSize - 1) / kLocalSize;

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "smith_visibility_crosscheck.comp",
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
			reinterpret_cast<u8 const *>(samples.data()),
			count * sizeof(f32v3)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v2),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	GgxIsoPush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups);

	auto out = test::readback<f32v2>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

std::vector<f32v2> evaluate_smith_g1_crosscheck(std::vector<f32v2> const & samples) {
	u32 const count = (u32)samples.size();
	static constexpr u32 kLocalSize = 64;
	u32 const groups = (count + kLocalSize - 1) / kLocalSize;

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "smith_g1_crosscheck.comp",
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

	GgxIsoPush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups);

	auto out = test::readback<f32v2>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("microfacet: ggx isotropic golden values at normal and grazing") {
	std::vector<f32> const alphas = { 0.05f, 0.1f, 0.3f, 0.5f, 0.8f, 1.0f };

	std::vector<f32v2> samples;
	for (f32 alpha : alphas) {
		samples.push_back({ 1.0f, alpha }); // normal incidence, dotNorH=1
		samples.push_back({ 0.0f, alpha }); // grazing microfacet, dotNorH=0
	}
	auto const result = evaluate_ggx_iso(samples);

	constexpr f64 kPi = 3.14159265358979323846;
	for (u32 i = 0; i < alphas.size(); ++i) {
		f32 const alpha = alphas[i];
		f64 const expectedNormal = 1.0 / (kPi * alpha * alpha);
		f64 const expectedGrazing = (f64)(alpha * alpha) / kPi;
		CAPTURE(alpha);
		CHECK(result[i * 2 + 0] == doctest::Approx(expectedNormal).epsilon(0.001));
		CHECK(result[i * 2 + 1] == doctest::Approx(expectedGrazing).epsilon(0.001));
	}
}

TEST_CASE("microfacet: ggx isotropic is non-negative and monotonic decreasing off-normal") {
	std::vector<f32> const alphas = { 0.05f, 0.1f, 0.3f, 0.5f, 0.8f, 1.0f };

	for (f32 alpha : alphas) {
		std::vector<f32v2> samples;
		for (f32 ct = 1.0f; ct >= 0.0f; ct -= 0.01f) { samples.push_back({ ct, alpha }); }
		auto const result = evaluate_ggx_iso(samples);

		f32 prev = 1e30f;
		for (u32 i = 0; i < result.size(); ++i) {
			CAPTURE(alpha);
			CAPTURE(i);
			CHECK(result[i] >= 0.0f);
			// non-increasing as dotNorH decreases (moves off the normal)
			CHECK(result[i] <= prev * 1.0001f);
			prev = result[i];
		}
	}
}

TEST_CASE("microfacet: ggx isotropic normalization integrates to 1 over the hemisphere") {
	// integral_hemisphere D(h) * cosThetaH dOmega_h == 1 is the defining
	// property of a valid microfacet ndf. isotropic in phi, so
	// integral = 2*pi * integral_0^(pi/2) D(cosTheta) * cosTheta * sinTheta dTheta
	std::vector<f32> const alphas = { 0.2f, 0.4f, 0.6f, 0.8f };
	constexpr u32 kSteps = 20000;
	constexpr f64 kPi = 3.14159265358979323846;

	for (f32 alpha : alphas) {
		std::vector<f32v2> samples(kSteps);
		std::vector<f32> cosThetas(kSteps), sinThetas(kSteps);
		f64 const dTheta = (kPi / 2.0) / (f64)kSteps;
		for (u32 i = 0; i < kSteps; ++i) {
			f64 const theta = ((f64)i + 0.5) * dTheta;
			f32 const cosTheta = (f32)std::cos(theta);
			cosThetas[i] = cosTheta;
			sinThetas[i] = (f32)std::sin(theta);
			samples[i] = { cosTheta, alpha };
		}
		auto const dValues = evaluate_ggx_iso(samples);

		f64 integral = 0.0;
		for (u32 i = 0; i < kSteps; ++i) {
			integral += (f64)dValues[i] * cosThetas[i] * sinThetas[i] * dTheta;
		}
		integral *= 2.0 * kPi;

		CAPTURE(alpha);
		CAPTURE(integral);
		CHECK(integral == doctest::Approx(1.0).epsilon(0.01));
	}
}

TEST_CASE("microfacet: ggx anisotropic reduces to isotropic when alphaX == alphaY") {
	std::vector<f32> const alphas = { 0.1f, 0.3f, 0.5f, 0.9f };
	std::vector<f32> const dotNorHs = { 0.0f, 0.2f, 0.5f, 0.8f, 0.95f, 1.0f };

	std::vector<f32v2> isoSamples;
	std::vector<f32> anisoFlat;
	for (f32 alpha : alphas) {
		for (f32 dotNorH : dotNorHs) {
			isoSamples.push_back({ dotNorH, alpha });
			f32 const s = std::sqrt(std::max(0.0f, 1.0f - dotNorH * dotNorH));
			push_aniso_sample(anisoFlat, { s, 0.0f, dotNorH }, { alpha, alpha });
		}
	}
	auto const isoResult = evaluate_ggx_iso(isoSamples);
	auto const anisoResult = evaluate_ggx_aniso(anisoFlat);

	for (u32 i = 0; i < isoResult.size(); ++i) {
		CAPTURE(i);
		CHECK(anisoResult[i] == doctest::Approx(isoResult[i]).epsilon(0.001));
	}
}

TEST_CASE("microfacet: ggx anisotropic is non-negative (fuzz)") {
	std::vector<f32> flat;
	u32 seed = 999u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 1024u; ++i) {
		f32 const hx = next() * 2.0f - 1.0f;
		f32 const hy = next() * 2.0f - 1.0f;
		f32 const hz = std::sqrt(std::max(0.0f, 1.0f - hx * hx - hy * hy));
		push_aniso_sample(
			flat, { hx, hy, hz },
			{ 0.02f + next() * 0.98f, 0.02f + next() * 0.98f }
		);
	}
	auto const result = evaluate_ggx_aniso(flat);

	for (f32 v : result) { CHECK(v >= 0.0f); }
}

TEST_CASE("microfacet: ggx anisotropic normalization integrates to 1 (stretched alpha)") {
	constexpr f32v2 kAlpha = { 0.3f, 0.5f };
	constexpr u32 kThetaSteps = 400;
	constexpr u32 kPhiSteps = 200;
	constexpr f64 kPi = 3.14159265358979323846;

	std::vector<f32> flat;
	std::vector<f32> weights;
	flat.reserve((size_t)kThetaSteps * kPhiSteps * 5);
	weights.reserve((size_t)kThetaSteps * kPhiSteps);

	f64 const dTheta = (kPi / 2.0) / (f64)kThetaSteps;
	f64 const dPhi = (2.0 * kPi) / (f64)kPhiSteps;
	for (u32 ti = 0; ti < kThetaSteps; ++ti) {
		f64 const theta = ((f64)ti + 0.5) * dTheta;
		f32 const cosTheta = (f32)std::cos(theta);
		f32 const sinTheta = (f32)std::sin(theta);
		for (u32 pi = 0; pi < kPhiSteps; ++pi) {
			f64 const phi = ((f64)pi + 0.5) * dPhi;
			f32v3 const h {
				sinTheta * (f32)std::cos(phi),
				sinTheta * (f32)std::sin(phi),
				cosTheta,
			};
			push_aniso_sample(flat, h, kAlpha);
			weights.push_back(cosTheta * sinTheta * (f32)(dTheta * dPhi));
		}
	}
	auto const dValues = evaluate_ggx_aniso(flat);

	f64 integral = 0.0;
	for (u32 i = 0; i < dValues.size(); ++i) {
		integral += (f64)dValues[i] * (f64)weights[i];
	}

	CAPTURE(integral);
	CHECK(integral == doctest::Approx(1.0).epsilon(0.02));
}

TEST_CASE("microfacet: ggx anisotropic matches materialx's mx_ggx_NDF") {
	std::vector<f32> flat;
	u32 seed = 42u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 2048u; ++i) {
		f32 const hx = next() * 2.0f - 1.0f;
		f32 const hy = next() * 2.0f - 1.0f;
		f32 const hz = std::sqrt(std::max(0.0f, 1.0f - hx * hx - hy * hy));
		push_aniso_sample(
			flat, { hx, hy, hz },
			{ 0.02f + next() * 0.98f, 0.02f + next() * 0.98f }
		);
	}
	auto const result = evaluate_ggx_aniso_crosscheck(flat);

	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(result[i].x == doctest::Approx(result[i].y).epsilon(0.001));
	}
}

TEST_CASE("microfacet: ggx isotropic matches materialx directly (equal-alpha aniso call)") {
	// the aniso-reduces-to-iso test above only proves internal consistency
	// between my own two functions; this compares utilMicrofacetGgxDistribution
	// against materialx's mx_ggx_NDF(H, (alpha,alpha)) directly, with no hop
	// through my own aniso function on either side
	std::vector<f32> const alphas = { 0.05f, 0.1f, 0.2f, 0.35f, 0.5f, 0.7f, 0.9f, 1.0f };

	std::vector<f32v2> isoSamples;
	std::vector<f32> anisoFlat;
	for (f32 alpha : alphas) {
		for (f32 dotNorH = 0.0f; dotNorH <= 1.0f; dotNorH += 0.02f) {
			isoSamples.push_back({ dotNorH, alpha });
			f32 const s = std::sqrt(std::max(0.0f, 1.0f - dotNorH * dotNorH));
			push_aniso_sample(anisoFlat, { s, 0.0f, dotNorH }, { alpha, alpha });
		}
	}
	auto const isoResult = evaluate_ggx_iso(isoSamples);
	auto const crosscheckResult = evaluate_ggx_aniso_crosscheck(anisoFlat);

	for (u32 i = 0; i < isoResult.size(); ++i) {
		CAPTURE(i);
		// .y is materialx's mx_ggx_NDF; .x (my own aniso) is intentionally
		// unused here
		CHECK(isoResult[i] == doctest::Approx(crosscheckResult[i].y).epsilon(0.001));
	}
}

TEST_CASE("microfacet: ggx isotropic heatmap (dotNorH x alpha)") {
	// x: dotNorH, 1 (normal) -> 0 (grazing). y: alpha, 0.05 -> 1.0
	// tonemapped via v/(v+1) since raw D spans a huge dynamic range
	constexpr u32 kWidth = 512;
	constexpr u32 kHeight = 512;
	constexpr f32 kAlphaMin = 0.05f;
	constexpr f32 kAlphaMax = 1.0f;

	std::vector<f32v2> samples(kWidth * kHeight);
	for (u32 y = 0; y < kHeight; ++y) {
		f32 const alpha = kAlphaMin + (kAlphaMax - kAlphaMin) * (f32)y / (f32)(kHeight - 1);
		for (u32 x = 0; x < kWidth; ++x) {
			f32 const dotNorH = 1.0f - (f32)x / (f32)(kWidth - 1);
			samples[y * kWidth + x] = { dotNorH, alpha };
		}
	}
	auto const result = evaluate_ggx_iso(samples);

	std::vector<f32> tonemapped(result.size());
	for (u32 i = 0; i < result.size(); ++i) {
		tonemapped[i] = result[i] / (result[i] + 1.0f);
	}

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		tonemapped, tonemapped, tonemapped, kWidth, kHeight,
		MICROFACET_OUTPUT_DIR "ggx_isotropic_heatmap.png", &nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);
}

namespace {

void write_ggx_footprint(f32v2 alpha, char const * path) {
	// tangent-plane slice: x=hx, y=hy over the unit disk. outside the disk
	// (invalid H) renders as background black.
	constexpr u32 kSize = 512;

	std::vector<f32> flat;
	std::vector<bool> valid(kSize * kSize, false);
	flat.reserve((size_t)kSize * kSize * 5);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const hy = 2.0f * (f32)y / (f32)(kSize - 1) - 1.0f;
		for (u32 x = 0; x < kSize; ++x) {
			f32 const hx = 2.0f * (f32)x / (f32)(kSize - 1) - 1.0f;
			f32 const r2 = hx * hx + hy * hy;
			bool const inside = r2 <= 1.0f;
			valid[y * kSize + x] = inside;
			f32 const hz = inside ? std::sqrt(1.0f - r2) : 1.0f;
			push_aniso_sample(flat, { hx, hy, hz }, alpha);
		}
	}
	auto const result = evaluate_ggx_aniso(flat);

	std::vector<f32> tonemapped(result.size(), 0.0f);
	for (u32 i = 0; i < result.size(); ++i) {
		if (valid[i]) { tonemapped[i] = result[i] / (result[i] + 1.0f); }
	}

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		tonemapped, tonemapped, tonemapped, kSize, kSize, path, &nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);
}

} // namespace

TEST_CASE("microfacet: ggx anisotropic footprint, stretched") {
	write_ggx_footprint(
		{ 0.08f, 0.35f }, MICROFACET_OUTPUT_DIR "ggx_footprint_anisotropic.png"
	);
}

TEST_CASE("microfacet: ggx anisotropic footprint, isotropic control") {
	write_ggx_footprint(
		{ 0.2f, 0.2f }, MICROFACET_OUTPUT_DIR "ggx_footprint_isotropic_control.png"
	);
}

TEST_CASE("smith visibility: zero for backfacing directions") {
	std::vector<f32v3> const samples = {
		{ -0.1f, 0.5f, 0.3f }, { 0.5f, -0.1f, 0.3f }, { 0.0f, 0.5f, 0.3f },
		{ 0.5f, 0.0f, 0.3f }, { -0.5f, -0.5f, 0.3f },
	};
	auto const result = evaluate_smith_visibility(samples);
	for (f32 v : result) { CHECK(v == 0.0f); }
}

TEST_CASE("smith visibility: golden value 0.25 at normal incidence, any alpha") {
	std::vector<f32v3> samples;
	std::vector<f32> const alphas = { 0.05f, 0.2f, 0.5f, 0.8f, 1.0f };
	for (f32 alpha : alphas) { samples.push_back({ 1.0f, 1.0f, alpha }); }
	auto const result = evaluate_smith_visibility(samples);

	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(alphas[i]);
		CHECK(result[i] == doctest::Approx(0.25).epsilon(0.001));
	}
}

TEST_CASE("smith visibility: symmetric under swapping wi/wo (fuzz)") {
	std::vector<f32v3> forward, swapped;
	u32 seed = 7u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 1024u; ++i) {
		f32 const wi = 0.01f + next() * 0.98f;
		f32 const wo = 0.01f + next() * 0.98f;
		f32 const alpha = 0.02f + next() * 0.98f;
		forward.push_back({ wi, wo, alpha });
		swapped.push_back({ wo, wi, alpha });
	}
	auto const fwd = evaluate_smith_visibility(forward);
	auto const swp = evaluate_smith_visibility(swapped);

	for (u32 i = 0; i < fwd.size(); ++i) {
		CAPTURE(i);
		CHECK(fwd[i] == doctest::Approx(swp[i]).epsilon(0.0001));
	}
}

TEST_CASE("smith visibility: non-negative and finite (fuzz)") {
	// no finite upper bound to assert: at alpha=1, V = 0.5/(dotNorWi+dotNorWo),
	// which is unbounded as both cosines approach grazing. 0.25 (checked
	// above) is the value at normal incidence specifically, not a max.
	std::vector<f32v3> samples;
	u32 seed = 123u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 1024u; ++i) {
		samples.push_back({
			0.01f + next() * 0.98f, 0.01f + next() * 0.98f, 0.02f + next() * 0.98f
		});
	}
	auto const result = evaluate_smith_visibility(samples);
	for (f32 v : result) {
		CHECK(v >= 0.0f);
		CHECK(std::isfinite(v));
	}
}

TEST_CASE("smith visibility: matches materialx's mx_ggx_smith_G2 (folded)") {
	std::vector<f32v3> samples;
	u32 seed = 55u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 2048u; ++i) {
		samples.push_back({
			0.01f + next() * 0.98f, 0.01f + next() * 0.98f, 0.02f + next() * 0.98f
		});
	}
	auto const result = evaluate_smith_visibility_crosscheck(samples);
	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(result[i].x == doctest::Approx(result[i].y).epsilon(0.001));
	}
}

TEST_CASE("smith g1: golden values at normal and grazing incidence") {
	std::vector<f32> const alphas = { 0.05f, 0.2f, 0.5f, 0.8f, 1.0f };
	std::vector<f32v2> samples;
	for (f32 alpha : alphas) {
		samples.push_back({ 1.0f, alpha }); // normal: G1 == 1
		samples.push_back({ 0.0f, alpha }); // grazing: G1 == 0
	}
	auto const result = evaluate_smith_g1(samples);
	for (u32 i = 0; i < alphas.size(); ++i) {
		CAPTURE(alphas[i]);
		CHECK(result[i * 2 + 0] == doctest::Approx(1.0).epsilon(0.001));
		CHECK(result[i * 2 + 1] == doctest::Approx(0.0).epsilon(0.001));
	}
}

TEST_CASE("smith g1: bounded in [0,1] (fuzz)") {
	std::vector<f32v2> samples;
	u32 seed = 321u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 1024u; ++i) {
		samples.push_back({ next(), 0.02f + next() * 0.98f });
	}
	auto const result = evaluate_smith_g1(samples);
	for (f32 v : result) {
		CHECK(v >= -0.0001f);
		CHECK(v <= 1.0001f);
	}
}

TEST_CASE("smith g1: matches materialx's mx_ggx_smith_G1") {
	std::vector<f32v2> samples;
	u32 seed = 654u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 1024u; ++i) {
		samples.push_back({ 0.01f + next() * 0.98f, 0.02f + next() * 0.98f });
	}
	auto const result = evaluate_smith_g1_crosscheck(samples);
	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(result[i].x == doctest::Approx(result[i].y).epsilon(0.001));
	}
}

TEST_CASE("smith visibility aniso: zero for backfacing directions") {
	std::vector<f32> flat;
	auto push8 = [&](f32v3 wi, f32v3 wo, f32v2 alpha) {
		flat.push_back(wi.x); flat.push_back(wi.y); flat.push_back(wi.z);
		flat.push_back(wo.x); flat.push_back(wo.y); flat.push_back(wo.z);
		flat.push_back(alpha.x); flat.push_back(alpha.y);
	};
	push8({ 0.0f, 0.0f, -0.1f }, { 0.0f, 0.0f, 0.5f }, { 0.3f, 0.4f });
	push8({ 0.0f, 0.0f, 0.5f }, { 0.0f, 0.0f, -0.1f }, { 0.3f, 0.4f });
	push8({ 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.5f }, { 0.3f, 0.4f });

	auto const result = evaluate_smith_visibility_aniso(flat);
	for (f32 v : result) { CHECK(v == 0.0f); }
}

TEST_CASE("smith visibility aniso: reduces to isotropic when alphaX == alphaY") {
	std::vector<f32> const alphas = { 0.1f, 0.3f, 0.6f, 0.9f };
	std::vector<f32> const cosines = { 0.05f, 0.2f, 0.5f, 0.8f, 0.99f };

	std::vector<f32v3> isoSamples;
	std::vector<f32> anisoFlat;
	auto push8 = [&](f32v3 wi, f32v3 wo, f32v2 alpha) {
		anisoFlat.push_back(wi.x); anisoFlat.push_back(wi.y); anisoFlat.push_back(wi.z);
		anisoFlat.push_back(wo.x); anisoFlat.push_back(wo.y); anisoFlat.push_back(wo.z);
		anisoFlat.push_back(alpha.x); anisoFlat.push_back(alpha.y);
	};
	for (f32 alpha : alphas) {
		for (f32 dotWi : cosines) {
			for (f32 dotWo : cosines) {
				isoSamples.push_back({ dotWi, dotWo, alpha });
				f32 const sWi = std::sqrt(std::max(0.0f, 1.0f - dotWi * dotWi));
				f32 const sWo = std::sqrt(std::max(0.0f, 1.0f - dotWo * dotWo));
				push8({ sWi, 0.0f, dotWi }, { sWo, 0.0f, dotWo }, { alpha, alpha });
			}
		}
	}
	auto const isoResult = evaluate_smith_visibility(isoSamples);
	auto const anisoResult = evaluate_smith_visibility_aniso(anisoFlat);

	for (u32 i = 0; i < isoResult.size(); ++i) {
		CAPTURE(i);
		CHECK(anisoResult[i] == doctest::Approx(isoResult[i]).epsilon(0.001));
	}
}

TEST_CASE("smith visibility aniso: symmetric under swapping wi/wo (fuzz)") {
	std::vector<f32> forward, swapped;
	u32 seed = 88u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	auto push8 = [](std::vector<f32> & flat, f32v3 wi, f32v3 wo, f32v2 alpha) {
		flat.push_back(wi.x); flat.push_back(wi.y); flat.push_back(wi.z);
		flat.push_back(wo.x); flat.push_back(wo.y); flat.push_back(wo.z);
		flat.push_back(alpha.x); flat.push_back(alpha.y);
	};
	for (u32 i = 0; i < 512u; ++i) {
		f32 const hx1 = next() * 2.0f - 1.0f, hy1 = next() * 2.0f - 1.0f;
		f32 const hz1 = std::sqrt(std::max(0.0f, 1.0f - hx1*hx1 - hy1*hy1));
		f32 const hx2 = next() * 2.0f - 1.0f, hy2 = next() * 2.0f - 1.0f;
		f32 const hz2 = std::sqrt(std::max(0.0f, 1.0f - hx2*hx2 - hy2*hy2));
		f32v2 const alpha = { 0.02f + next() * 0.98f, 0.02f + next() * 0.98f };
		push8(forward, { hx1, hy1, hz1 }, { hx2, hy2, hz2 }, alpha);
		push8(swapped, { hx2, hy2, hz2 }, { hx1, hy1, hz1 }, alpha);
	}
	auto const fwd = evaluate_smith_visibility_aniso(forward);
	auto const swp = evaluate_smith_visibility_aniso(swapped);

	for (u32 i = 0; i < fwd.size(); ++i) {
		CAPTURE(i);
		CHECK(fwd[i] == doctest::Approx(swp[i]).epsilon(0.0001));
	}
}

TEST_CASE("smith visibility aniso: non-negative (fuzz)") {
	std::vector<f32> flat;
	u32 seed = 99u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	auto push8 = [](std::vector<f32> & f, f32v3 wi, f32v3 wo, f32v2 alpha) {
		f.push_back(wi.x); f.push_back(wi.y); f.push_back(wi.z);
		f.push_back(wo.x); f.push_back(wo.y); f.push_back(wo.z);
		f.push_back(alpha.x); f.push_back(alpha.y);
	};
	for (u32 i = 0; i < 1024u; ++i) {
		f32 const hx1 = next() * 2.0f - 1.0f, hy1 = next() * 2.0f - 1.0f;
		f32 const hz1 = std::sqrt(std::max(0.0f, 1.0f - hx1*hx1 - hy1*hy1));
		f32 const hx2 = next() * 2.0f - 1.0f, hy2 = next() * 2.0f - 1.0f;
		f32 const hz2 = std::sqrt(std::max(0.0f, 1.0f - hx2*hx2 - hy2*hy2));
		push8(
			flat, { hx1, hy1, hz1 }, { hx2, hy2, hz2 },
			{ 0.02f + next() * 0.98f, 0.02f + next() * 0.98f }
		);
	}
	auto const result = evaluate_smith_visibility_aniso(flat);
	for (f32 v : result) { CHECK(v >= 0.0f); }
}

TEST_CASE("smith visibility: heatmap (dotNorWi x dotNorWo), symmetric across diagonal") {
	constexpr u32 kSize = 512;
	constexpr f32 kAlpha = 0.3f;

	std::vector<f32v3> samples(kSize * kSize);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const dotNorWo = (f32)y / (f32)(kSize - 1);
		for (u32 x = 0; x < kSize; ++x) {
			f32 const dotNorWi = (f32)x / (f32)(kSize - 1);
			samples[y * kSize + x] = { dotNorWi, dotNorWo, kAlpha };
		}
	}
	auto const result = evaluate_smith_visibility(samples);

	std::vector<f32> tonemapped(result.size());
	for (u32 i = 0; i < result.size(); ++i) {
		tonemapped[i] = result[i] / (result[i] + 0.25f); // 0.25 is the max value
	}

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		tonemapped, tonemapped, tonemapped, kSize, kSize,
		MICROFACET_OUTPUT_DIR "smith_visibility_heatmap.png", &nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);
}

TEST_CASE("smith g1: heatmap (dotNorWi x alpha)") {
	constexpr u32 kWidth = 512;
	constexpr u32 kHeight = 512;

	std::vector<f32v2> samples(kWidth * kHeight);
	for (u32 y = 0; y < kHeight; ++y) {
		f32 const alpha = 0.02f + 0.98f * (f32)y / (f32)(kHeight - 1);
		for (u32 x = 0; x < kWidth; ++x) {
			f32 const dotNorWi = (f32)x / (f32)(kWidth - 1);
			samples[y * kWidth + x] = { dotNorWi, alpha };
		}
	}
	auto const result = evaluate_smith_g1(samples);

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		result, result, result, kWidth, kHeight,
		MICROFACET_OUTPUT_DIR "smith_g1_heatmap.png", &nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);
}

} // TEST_SUITE("[headless]")
