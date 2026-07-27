#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <ponder/zeltner-tables.hpp>
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

// tests for util-material-openpbr-ibl.glsl: the split-sum (N=V=R) raster
// counterparts of the path-traced openPbrEvaluateF lobe stack. conductor
// and coat each get a closed-form-vs-golden-hemisphere-integral crosscheck
// (golden = numeric quadrature of the real path-traced eval function,
// coat_ibl_furnace_integrate.comp / reused metallic_lobe_furnace_integrate.comp,
// not copied from the closed form under test). fuzz's attenuation term
// reuses the already-validated zeltner lookup verbatim (test-fuzz-eval.cpp),
// so it isn't re-crosschecked here; only the full-stack wiring is.

namespace {

static constexpr u32 kLocalSize = 64;
static u32 groups_for(u32 count) { return (count + kLocalSize - 1) / kLocalSize; }

// -----------------------------------------------------------------------------
// -- conductor: openPbrConductorIblAlbedo vs golden hemisphere integral
// -----------------------------------------------------------------------------

void push_metal_sample(
	std::vector<f32> & flat,
	f32 muI, f32 specularRoughness, f32 specularRoughnessAnisotropy,
	f32 baseWeight, f32 baseColorScalar, f32 specularWeight,
	f32 specularColorScalar
) {
	flat.push_back(muI);
	flat.push_back(specularRoughness);
	flat.push_back(specularRoughnessAnisotropy);
	flat.push_back(baseWeight);
	flat.push_back(baseColorScalar);
	flat.push_back(specularWeight);
	flat.push_back(specularColorScalar);
	flat.push_back(0.0f);
}

std::vector<f32> integrate_metal_furnace(
	std::vector<f32> const & samplesFlat, u32 thetaSteps, u32 phiSteps
) {
	u32 const count = (u32)samplesFlat.size() / 8u;
	REQUIRE(samplesFlat.size() == (size_t)count * 8u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "metallic_lobe_furnace_integrate.comp",
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

std::vector<f32v3> evaluate_conductor_ibl_albedo(std::vector<f32> const & samplesFlat) {
	u32 const count = (u32)samplesFlat.size() / 8u;
	REQUIRE(samplesFlat.size() == (size_t)count * 8u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "conductor_ibl_albedo_evaluate.comp",
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

// -----------------------------------------------------------------------------
// -- conductor + thin-film: openPbrConductorIblAlbedo vs golden hemisphere
// -- integral. unlike plain f82, the thin-film term is evaluated at
// -- dot(h, wi) in the real eval, not dot(nor, wi) -- the closed form
// -- substitutes mu (the N=V=R mirror-direction value) for it, so this is
// -- expected to diverge more from golden than the plain-metal crosscheck,
// -- especially at higher roughness (wider lobe, dot(h,wi) strays further
// -- from mu across it)
// -----------------------------------------------------------------------------

void push_thinfilm_metal_sample(
	std::vector<f32> & flat,
	f32 muI, f32 specularRoughness, f32 specularRoughnessAnisotropy,
	f32 baseWeight, f32 baseColorScalar, f32 specularWeight,
	f32 specularColorScalar, f32 thinFilmWeight, f32 thinFilmThickness,
	f32 thinFilmIor
) {
	flat.push_back(muI);
	flat.push_back(specularRoughness);
	flat.push_back(specularRoughnessAnisotropy);
	flat.push_back(baseWeight);
	flat.push_back(baseColorScalar);
	flat.push_back(specularWeight);
	flat.push_back(specularColorScalar);
	flat.push_back(thinFilmWeight);
	flat.push_back(thinFilmThickness);
	flat.push_back(thinFilmIor);
}

std::vector<f32v3> integrate_thinfilm_metal_furnace(
	std::vector<f32> const & samplesFlat, u32 thetaSteps, u32 phiSteps
) {
	u32 const count = (u32)samplesFlat.size() / 10u;
	REQUIRE(samplesFlat.size() == (size_t)count * 10u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "thinfilm_metallic_furnace_integrate.comp",
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

	struct Push { u64 inVa; u64 outVa; u32 thetaSteps; u32 phiSteps; u32 count; };
	Push const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.thetaSteps = thetaSteps,
		.phiSteps = phiSteps,
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	auto out = test::readback<f32v3>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

std::vector<f32v3> evaluate_conductor_ibl_albedo_thinfilm(
	std::vector<f32> const & samplesFlat
) {
	u32 const count = (u32)samplesFlat.size() / 10u;
	REQUIRE(samplesFlat.size() == (size_t)count * 10u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "conductor_ibl_albedo_thinfilm_evaluate.comp",
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

// -----------------------------------------------------------------------------
// -- coat: openPbrCoatIblEvaluate vs golden hemisphere integral
// -----------------------------------------------------------------------------

void push_coat_sample(
	std::vector<f32> & flat,
	f32 muI, f32 coatWeight, f32 coatRoughness, f32 coatIor, f32 coatDarkening,
	f32 coatColorScalar, f32 specularRoughness, f32 specularWeight,
	f32 specularIor, f32 baseMetalness, f32 baseColorScalar
) {
	flat.push_back(muI);
	flat.push_back(coatWeight);
	flat.push_back(coatRoughness);
	flat.push_back(coatIor);
	flat.push_back(coatDarkening);
	flat.push_back(coatColorScalar);
	flat.push_back(specularRoughness);
	flat.push_back(specularWeight);
	flat.push_back(specularIor);
	flat.push_back(baseMetalness);
	flat.push_back(baseColorScalar);
}

std::vector<f32> integrate_coat_furnace(
	std::vector<f32> const & samplesFlat, u32 thetaSteps, u32 phiSteps
) {
	u32 const count = (u32)samplesFlat.size() / 11u;
	REQUIRE(samplesFlat.size() == (size_t)count * 11u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "coat_ibl_furnace_integrate.comp",
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

std::vector<f32v3> evaluate_coat_ibl(std::vector<f32> const & samplesFlat) {
	u32 const count = (u32)samplesFlat.size() / 11u;
	REQUIRE(samplesFlat.size() == (size_t)count * 11u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "coat_ibl_evaluate.comp",
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

// -----------------------------------------------------------------------------
// -- full stack: openPbrIblEvaluateF
// -----------------------------------------------------------------------------

struct FullStackSample {
	f32 muI, dotCoatNorWi, irradianceScalar, prefilteredBaseScalar, prefilteredCoatScalar;
	f32 baseWeight, baseColorScalar, baseMetalness;
	f32 specularWeight, specularColorScalar, specularRoughness, specularIor;
	f32 coatWeight, coatRoughness, coatIor, coatDarkening, coatColorScalar;
	f32 fuzzWeight, fuzzColorScalar, fuzzRoughness;
};

void push_full_stack_sample(std::vector<f32> & flat, FullStackSample const & s) {
	flat.push_back(s.muI);
	flat.push_back(s.dotCoatNorWi);
	flat.push_back(s.irradianceScalar);
	flat.push_back(s.prefilteredBaseScalar);
	flat.push_back(s.prefilteredCoatScalar);
	flat.push_back(s.baseWeight);
	flat.push_back(s.baseColorScalar);
	flat.push_back(s.baseMetalness);
	flat.push_back(s.specularWeight);
	flat.push_back(s.specularColorScalar);
	flat.push_back(s.specularRoughness);
	flat.push_back(s.specularIor);
	flat.push_back(s.coatWeight);
	flat.push_back(s.coatRoughness);
	flat.push_back(s.coatIor);
	flat.push_back(s.coatDarkening);
	flat.push_back(s.coatColorScalar);
	flat.push_back(s.fuzzWeight);
	flat.push_back(s.fuzzColorScalar);
	flat.push_back(s.fuzzRoughness);
}

std::vector<f32v3> evaluate_full_stack(
	u32 zeltnerLtcParamHandle, std::vector<f32> const & samplesFlat
) {
	u32 const count = (u32)samplesFlat.size() / 20u;
	REQUIRE(samplesFlat.size() == (size_t)count * 20u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "openpbr_ibl_evaluate.comp",
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

	struct Push { u64 inVa; u64 outVa; u32 zeltnerLtcParamHandle; u32 count; };
	Push const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.zeltnerLtcParamHandle = zeltnerLtcParamHandle,
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	auto out = test::readback<f32v3>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

FullStackSample base_sample() {
	return FullStackSample {
		.muI = 0.5f, .dotCoatNorWi = 0.5f,
		.irradianceScalar = 1.0f, .prefilteredBaseScalar = 1.0f,
		.prefilteredCoatScalar = 1.0f,
		.baseWeight = 1.0f, .baseColorScalar = 1.0f, .baseMetalness = 0.0f,
		.specularWeight = 1.0f, .specularColorScalar = 1.0f,
		.specularRoughness = 0.5f, .specularIor = 1.5f,
		.coatWeight = 0.0f, .coatRoughness = 0.3f, .coatIor = 1.5f,
		.coatDarkening = 1.0f, .coatColorScalar = 1.0f,
		.fuzzWeight = 0.0f, .fuzzColorScalar = 1.0f, .fuzzRoughness = 0.3f,
	};
}

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("openpbr ibl: conductor closed form matches golden hemisphere integral") {
	// roughness floor 0.1 and phi=400, matching the established safe
	// quadrature range from "metallic lobe: white furnace == 1 across
	// roughness x incidence" -- below that the near-mirror lobe is only a
	// couple of quadrature samples wide and the integral itself becomes
	// the error source, not the lobe
	std::vector<f32> samples;
	for (f32 mu = 0.1f; mu <= 0.95f; mu += 0.15f) {
		for (f32 roughness = 0.1f; roughness <= 0.95f; roughness += 0.15f) {
			push_metal_sample(samples, mu, roughness, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f);
		}
	}
	auto const golden = integrate_metal_furnace(samples, 300u, 400u);
	auto const closedForm = evaluate_conductor_ibl_albedo(samples);
	REQUIRE(golden.size() == closedForm.size());

	f32 maxDeviation = 0.0f, sumDeviation = 0.0f;
	for (u32 i = 0; i < golden.size(); ++i) {
		f32 const deviation = std::abs(closedForm[i].x - golden[i]);
		CAPTURE(i);
		CAPTURE(golden[i]);
		CAPTURE(closedForm[i].x);
		maxDeviation = std::max(maxDeviation, deviation);
		sumDeviation += deviation;
	}
	f32 const meanDeviation = sumDeviation / (f32)golden.size();
	CAPTURE(maxDeviation);
	CAPTURE(meanDeviation);
	CHECK(meanDeviation < 0.02f);
	CHECK(maxDeviation < 0.06f);
}

TEST_CASE("openpbr ibl: conductor closed form matches golden integral (colored metal)") {
	std::vector<f32> samples;
	for (f32 mu = 0.05f; mu <= 0.95f; mu += 0.1f) {
		push_metal_sample(samples, mu, 0.4f, 0.0f, 1.0f, 0.6f, 1.0f, 0.9f);
	}
	auto const golden = integrate_metal_furnace(samples, 300u, 400u);
	auto const closedForm = evaluate_conductor_ibl_albedo(samples);

	f32 maxDeviation = 0.0f;
	for (u32 i = 0; i < golden.size(); ++i) {
		maxDeviation = std::max(maxDeviation, std::abs(closedForm[i].x - golden[i]));
	}
	CAPTURE(maxDeviation);
	CHECK(maxDeviation < 0.06f);
}

TEST_CASE("openpbr ibl: thin-film conductor closed form roughly matches golden integral at low roughness") {
	// low roughness: a narrow lobe means most reflected energy comes from
	// near the mirror direction, where the closed form's mu-for-dot(h,wi)
	// substitution is close to exact -- this is not expected to hold at
	// high roughness (see the golden shader's own comment)
	std::vector<f32> samples;
	for (f32 mu = 0.1f; mu <= 0.95f; mu += 0.15f) {
		for (f32 thickness : { 0.3f, 0.8f }) {
			push_thinfilm_metal_sample(
				samples, mu, /*specularRoughness=*/ 0.15f, 0.0f,
				/*baseWeight=*/ 1.0f, /*baseColorScalar=*/ 0.9f,
				/*specularWeight=*/ 1.0f, /*specularColorScalar=*/ 1.0f,
				/*thinFilmWeight=*/ 1.0f, thickness, /*thinFilmIor=*/ 1.4f
			);
		}
	}
	auto const golden = integrate_thinfilm_metal_furnace(samples, 300u, 400u);
	auto const closedForm = evaluate_conductor_ibl_albedo_thinfilm(samples);
	REQUIRE(golden.size() == closedForm.size());

	f32 maxDeviation = 0.0f, sumDeviation = 0.0f;
	for (u32 i = 0; i < golden.size(); ++i) {
		f32v3 const diff = {
			closedForm[i].x - golden[i].x,
			closedForm[i].y - golden[i].y,
			closedForm[i].z - golden[i].z,
		};
		f32 const deviation = (
			std::max({ std::abs(diff.x), std::abs(diff.y), std::abs(diff.z) })
		);
		maxDeviation = std::max(maxDeviation, deviation);
		sumDeviation += deviation;
	}
	f32 const meanDeviation = sumDeviation / (f32)golden.size();
	CAPTURE(maxDeviation);
	CAPTURE(meanDeviation);
	CHECK(meanDeviation < 0.05f);
	CHECK(maxDeviation < 0.15f);
}

TEST_CASE("openpbr ibl: thin-film conductor finite and energy-bounded (fuzz)") {
	u32 seed = 11u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};

	std::vector<f32> samples;
	for (u32 i = 0; i < 512u; ++i) {
		push_thinfilm_metal_sample(
			samples,
			/*muI=*/ std::max(next(), 0.02f),
			/*specularRoughness=*/ std::max(next(), 0.02f),
			/*specularRoughnessAnisotropy=*/ 0.0f,
			/*baseWeight=*/ 1.0f, /*baseColorScalar=*/ 0.5f + 0.5f * next(),
			/*specularWeight=*/ 1.0f, /*specularColorScalar=*/ 1.0f,
			/*thinFilmWeight=*/ next(),
			/*thinFilmThickness=*/ 2.0f * next(),
			/*thinFilmIor=*/ 1.0f + 2.0f * next()
		);
	}
	auto const out = evaluate_conductor_ibl_albedo_thinfilm(samples);

	for (f32v3 const & v : out) {
		CAPTURE(v.x);
		CAPTURE(v.y);
		CAPTURE(v.z);
		CHECK(std::isfinite(v.x));
		CHECK(std::isfinite(v.y));
		CHECK(std::isfinite(v.z));
		// iridescence redistributes energy across channels, so a single
		// channel can modestly exceed 1 -- the fitted albedo terms
		// themselves already tolerate small overshoot elsewhere in this
		// suite; this just guards against a real blowup
		CHECK(v.x >= -0.01f);
		CHECK(v.x <= 1.5f);
	}
}

TEST_CASE("openpbr ibl: thin-film conductor heatmap (mu x thickness)") {
	constexpr u32 kSize = 256;
	std::vector<f32> samples;
	for (u32 y = 0; y < kSize; ++y) {
		f32 const thickness = 2.0f * (f32)y / (f32)(kSize - 1);
		for (u32 x = 0; x < kSize; ++x) {
			f32 const mu = std::max((f32)x / (f32)(kSize - 1), 1e-3f);
			push_thinfilm_metal_sample(
				samples, mu, /*specularRoughness=*/ 0.15f, 0.0f,
				/*baseWeight=*/ 1.0f, /*baseColorScalar=*/ 0.9f,
				/*specularWeight=*/ 1.0f, /*specularColorScalar=*/ 1.0f,
				/*thinFilmWeight=*/ 1.0f, thickness, /*thinFilmIor=*/ 1.4f
			);
		}
	}
	auto const result = evaluate_conductor_ibl_albedo_thinfilm(samples);

	std::vector<f32> r(result.size()), g(result.size()), b(result.size());
	for (u32 i = 0; i < result.size(); ++i) {
		r[i] = result[i].x; g[i] = result[i].y; b[i] = result[i].z;
	}
	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		r, g, b, kSize, kSize,
		OPENPBR_IBL_OUTPUT_DIR "conductor_ibl_thinfilm_heatmap.png", &nanCount
	);
	CHECK(ok);
	CHECK(nanCount == 0);
}

TEST_CASE("openpbr ibl: coat closed form matches golden hemisphere integral") {
	std::vector<f32> samples;
	for (f32 mu = 0.1f; mu <= 0.95f; mu += 0.2f) {
		for (f32 coatWeight = 0.25f; coatWeight <= 1.0f; coatWeight += 0.25f) {
			for (f32 coatRoughness = 0.1f; coatRoughness <= 0.95f; coatRoughness += 0.3f) {
				push_coat_sample(
					samples, mu, coatWeight, coatRoughness,
					/*coatIor=*/ 1.5f, /*coatDarkening=*/ 1.0f,
					/*coatColorScalar=*/ 1.0f, /*specularRoughness=*/ 0.5f,
					/*specularWeight=*/ 1.0f, /*specularIor=*/ 1.5f,
					/*baseMetalness=*/ 0.0f, /*baseColorScalar=*/ 1.0f
				);
			}
		}
	}
	auto const golden = integrate_coat_furnace(samples, 300u, 400u);
	auto const closedForm = evaluate_coat_ibl(samples);
	REQUIRE(golden.size() == closedForm.size());

	f32 maxDeviation = 0.0f, sumDeviation = 0.0f;
	for (u32 i = 0; i < golden.size(); ++i) {
		f32 const deviation = std::abs(closedForm[i].x - golden[i]);
		CAPTURE(i);
		CAPTURE(golden[i]);
		CAPTURE(closedForm[i].x);
		maxDeviation = std::max(maxDeviation, deviation);
		sumDeviation += deviation;
	}
	f32 const meanDeviation = sumDeviation / (f32)golden.size();
	CAPTURE(maxDeviation);
	CAPTURE(meanDeviation);
	CHECK(meanDeviation < 0.03f);
	// closed form reuses utilMicrofacetDielectricAlbedo's schlick-fit
	// albedo for the coat's own reflectance in place of the golden
	// reference's exact per-angle fresnel; residual gap is that fit's own
	// approximation error, not a coat-layering bug
	CHECK(maxDeviation < 0.18f);
}

TEST_CASE("openpbr ibl: coat closed form matches golden integral (tinted coat, over metal)") {
	std::vector<f32> samples;
	for (f32 mu = 0.1f; mu <= 0.95f; mu += 0.15f) {
		push_coat_sample(
			samples, mu, /*coatWeight=*/ 1.0f, /*coatRoughness=*/ 0.2f,
			/*coatIor=*/ 1.5f, /*coatDarkening=*/ 1.0f,
			/*coatColorScalar=*/ 0.7f, /*specularRoughness=*/ 0.3f,
			/*specularWeight=*/ 1.0f, /*specularIor=*/ 1.5f,
			/*baseMetalness=*/ 1.0f, /*baseColorScalar=*/ 0.9f
		);
	}
	auto const golden = integrate_coat_furnace(samples, 300u, 400u);
	auto const closedForm = evaluate_coat_ibl(samples);

	f32 maxDeviation = 0.0f;
	for (u32 i = 0; i < golden.size(); ++i) {
		maxDeviation = std::max(maxDeviation, std::abs(closedForm[i].x - golden[i]));
	}
	CAPTURE(maxDeviation);
	CHECK(maxDeviation < 0.18f);
}

TEST_CASE("openpbr ibl: full stack reduces to lambertian diffuse with everything else off") {
	auto sample = base_sample();
	sample.specularWeight = 0.0f;
	sample.baseMetalness = 0.0f;
	sample.coatWeight = 0.0f;
	sample.fuzzWeight = 0.0f;
	sample.baseColorScalar = 0.7f;
	sample.baseWeight = 0.8f;
	sample.irradianceScalar = 1.3f;

	std::vector<f32> samples;
	push_full_stack_sample(samples, sample);

	auto tables = ponder::zeltner_tables_create();
	auto const out = evaluate_full_stack(tables.zeltnerLtcParamHandle, samples);
	ponder::zeltner_tables_destroy(tables);

	f32 const expected = sample.baseColorScalar * sample.baseWeight * sample.irradianceScalar;
	CHECK(out[0].x == doctest::Approx(expected).epsilon(0.001));
	CHECK(out[0].y == doctest::Approx(expected).epsilon(0.001));
	CHECK(out[0].z == doctest::Approx(expected).epsilon(0.001));
}

TEST_CASE("openpbr ibl: fuzz term is actually wired in (changes output when enabled)") {
	auto tables = ponder::zeltner_tables_create();

	// a fully white furnace (irradiance = baseColor = fuzzColor = 1) makes
	// own + attenuation * base an energy-conserving identity (== 1)
	// regardless of the fuzz/base split, so this needs a non-white base to
	// actually distinguish "wired in" from "coincidentally cancels out"
	auto withoutFuzz = base_sample();
	withoutFuzz.baseColorScalar = 0.4f;
	withoutFuzz.fuzzWeight = 0.0f;
	auto withFuzz = base_sample();
	withFuzz.baseColorScalar = 0.4f;
	withFuzz.fuzzWeight = 1.0f;
	withFuzz.fuzzRoughness = 0.4f;
	withFuzz.fuzzColorScalar = 1.0f;

	std::vector<f32> samplesWithout, samplesWith;
	push_full_stack_sample(samplesWithout, withoutFuzz);
	push_full_stack_sample(samplesWith, withFuzz);

	auto const outWithout = evaluate_full_stack(tables.zeltnerLtcParamHandle, samplesWithout);
	auto const outWith = evaluate_full_stack(tables.zeltnerLtcParamHandle, samplesWith);
	ponder::zeltner_tables_destroy(tables);

	CHECK(std::abs(outWith[0].x - outWithout[0].x) > 1e-4f);
}

TEST_CASE("openpbr ibl: full stack finite and bounded under a uniform white furnace (fuzz)") {
	auto tables = ponder::zeltner_tables_create();

	u32 seed = 7u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};

	std::vector<f32> samples;
	u32 const kCount = 512u;
	for (u32 i = 0; i < kCount; ++i) {
		FullStackSample s {
			.muI = std::max(next(), 0.02f),
			.dotCoatNorWi = std::max(next(), 0.02f),
			.irradianceScalar = 1.0f,
			.prefilteredBaseScalar = 1.0f,
			.prefilteredCoatScalar = 1.0f,
			.baseWeight = 1.0f,
			.baseColorScalar = 1.0f,
			.baseMetalness = next(),
			.specularWeight = 1.0f,
			.specularColorScalar = 1.0f,
			.specularRoughness = std::max(next(), 0.02f),
			.specularIor = 1.2f + next(),
			.coatWeight = next(),
			.coatRoughness = std::max(next(), 0.02f),
			.coatIor = 1.2f + next(),
			.coatDarkening = next(),
			.coatColorScalar = 0.7f + 0.3f * next(),
			.fuzzWeight = next(),
			.fuzzColorScalar = 1.0f,
			.fuzzRoughness = std::max(next(), 0.02f),
		};
		push_full_stack_sample(samples, s);
	}

	auto const out = evaluate_full_stack(tables.zeltnerLtcParamHandle, samples);
	ponder::zeltner_tables_destroy(tables);

	for (f32v3 const & v : out) {
		CAPTURE(v.x);
		CAPTURE(v.y);
		CAPTURE(v.z);
		CHECK(std::isfinite(v.x));
		CHECK(std::isfinite(v.y));
		CHECK(std::isfinite(v.z));
		CHECK(v.x >= -0.01f);
		CHECK(v.x <= 1.5f);
	}
}

TEST_CASE("openpbr ibl: conductor albedo heatmap (mu x roughness, tinted metal)") {
	// baseColor < 1 (unlike a white furnace) so the f82 edge-tint dip is
	// actually visible instead of collapsing to a blank 1.0 everywhere
	constexpr u32 kSize = 256;
	std::vector<f32> samples;
	for (u32 y = 0; y < kSize; ++y) {
		f32 const roughness = std::max((f32)y / (f32)(kSize - 1), 1e-3f);
		for (u32 x = 0; x < kSize; ++x) {
			f32 const mu = std::max((f32)x / (f32)(kSize - 1), 1e-3f);
			push_metal_sample(samples, mu, roughness, 0.0f, 1.0f, 0.6f, 1.0f, 1.0f);
		}
	}
	auto const result = evaluate_conductor_ibl_albedo(samples);

	std::vector<f32> r(result.size()), g(result.size()), b(result.size());
	for (u32 i = 0; i < result.size(); ++i) {
		r[i] = result[i].x; g[i] = result[i].y; b[i] = result[i].z;
	}
	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		r, g, b, kSize, kSize,
		OPENPBR_IBL_OUTPUT_DIR "conductor_ibl_albedo_heatmap.png", &nanCount
	);
	CHECK(ok);
	CHECK(nanCount == 0);
}

TEST_CASE("openpbr ibl: coat attenuation heatmap (mu x coat weight)") {
	constexpr u32 kSize = 256;
	std::vector<f32> samples;
	for (u32 y = 0; y < kSize; ++y) {
		f32 const coatWeight = (f32)y / (f32)(kSize - 1);
		for (u32 x = 0; x < kSize; ++x) {
			f32 const mu = std::max((f32)x / (f32)(kSize - 1), 1e-3f);
			push_coat_sample(
				samples, mu, coatWeight, /*coatRoughness=*/ 0.3f,
				/*coatIor=*/ 1.5f, /*coatDarkening=*/ 1.0f,
				/*coatColorScalar=*/ 1.0f, /*specularRoughness=*/ 0.5f,
				/*specularWeight=*/ 1.0f, /*specularIor=*/ 1.5f,
				/*baseMetalness=*/ 0.0f, /*baseColorScalar=*/ 1.0f
			);
		}
	}
	auto const result = evaluate_coat_ibl(samples);

	std::vector<f32> r(result.size()), g(result.size()), b(result.size());
	for (u32 i = 0; i < result.size(); ++i) {
		r[i] = result[i].x; g[i] = result[i].y; b[i] = result[i].z;
	}
	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		r, g, b, kSize, kSize,
		OPENPBR_IBL_OUTPUT_DIR "coat_ibl_heatmap.png", &nanCount
	);
	CHECK(ok);
	CHECK(nanCount == 0);
}

} // TEST_SUITE("[headless]")
