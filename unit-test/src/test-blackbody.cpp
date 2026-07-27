#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <cmath>
#include <vector>

// tests for utilBlackbodyRadiance/utilCieXyz1931/utilPlanckRadiance
// (lib/ponder/shaders/util-blackbody.glsl), the vdb temperature-grid
// emission term added in the "physically-based vdb parameters" pass -- see
// lib/ponder/VDB_REFERENCES.md. golden/bounds checks plus a cpu f64
// reference transcription, matching this repo's established milestone
// coverage pattern (cf. test-thinfilm.cpp).

namespace {

static constexpr u32 kLocalSize = 64;
static u32 groups_for(u32 count) { return (count + kLocalSize - 1) / kLocalSize; }

// -----------------------------------------------------------------------------
// cpu f64 reference, transcribed independently from util-blackbody.glsl
// -----------------------------------------------------------------------------

struct RefRgb { f64 r, g, b; };

f64 ref_planck_radiance(f64 lambdaNm, f64 temperatureKelvin) {
	f64 const h = 6.62607015e-34;
	f64 const c = 2.99792458e8;
	f64 const kB = 1.380649e-23;
	f64 const lambdaM = lambdaNm * 1e-9;
	f64 const lambda5 = lambdaM * lambdaM * lambdaM * lambdaM * lambdaM;
	f64 const numer = 2.0 * h * c * c;
	f64 const expArg = (h * c) / (lambdaM * kB * std::max(temperatureKelvin, 1.0));
	return numer / (lambda5 * (std::exp(std::min(expArg, 80.0)) - 1.0));
}

RefRgb ref_cie_xyz_1931(f64 lambdaNm) {
	f64 const t1x = (
		(lambdaNm - 442.0) * (lambdaNm < 442.0 ? 0.0624 : 0.0374)
	);
	f64 const t2x = (
		(lambdaNm - 599.8) * (lambdaNm < 599.8 ? 0.0264 : 0.0323)
	);
	f64 const t3x = (
		(lambdaNm - 501.1) * (lambdaNm < 501.1 ? 0.0490 : 0.0382)
	);
	f64 const x = (
		0.362 * std::exp(-0.5 * t1x * t1x)
		+ 1.056 * std::exp(-0.5 * t2x * t2x)
		- 0.065 * std::exp(-0.5 * t3x * t3x)
	);

	f64 const t1y = (
		(lambdaNm - 568.8) * (lambdaNm < 568.8 ? 0.0213 : 0.0247)
	);
	f64 const t2y = (
		(lambdaNm - 530.9) * (lambdaNm < 530.9 ? 0.0613 : 0.0322)
	);
	f64 const y = (
		0.821 * std::exp(-0.5 * t1y * t1y)
		+ 0.286 * std::exp(-0.5 * t2y * t2y)
	);

	f64 const t1z = (
		(lambdaNm - 437.0) * (lambdaNm < 437.0 ? 0.0845 : 0.0278)
	);
	f64 const t2z = (
		(lambdaNm - 459.0) * (lambdaNm < 459.0 ? 0.0385 : 0.0725)
	);
	f64 const z = (
		1.217 * std::exp(-0.5 * t1z * t1z)
		+ 0.681 * std::exp(-0.5 * t2z * t2z)
	);

	return { x, y, z };
}

RefRgb ref_blackbody_radiance(f64 temperatureKelvin) {
	f64 const lambdaMin = 380.0;
	f64 const lambdaMax = 780.0;
	u32 const sampleCount = 32u;
	f64 const dLambdaNm = (lambdaMax - lambdaMin) / (f64)sampleCount;
	f64 const dLambdaM = dLambdaNm * 1e-9;

	f64 x = 0.0, y = 0.0, z = 0.0;
	for (u32 i = 0u; i < sampleCount; ++i) {
		f64 const lambdaNm = lambdaMin + ((f64)i + 0.5) * dLambdaNm;
		f64 const radiance = ref_planck_radiance(lambdaNm, temperatureKelvin);
		RefRgb const xyzAtLambda = ref_cie_xyz_1931(lambdaNm);
		x += xyzAtLambda.r * radiance * dLambdaM;
		y += xyzAtLambda.g * radiance * dLambdaM;
		z += xyzAtLambda.b * radiance * dLambdaM;
	}

	RefRgb const rgb {
		3.2406 * x - 1.5372 * y - 0.4986 * z,
		-0.9689 * x + 1.8758 * y + 0.0415 * z,
		0.0557 * x - 0.2040 * y + 1.0570 * z,
	};
	return {
		std::max(rgb.r, 0.0), std::max(rgb.g, 0.0), std::max(rgb.b, 0.0)
	};
}

// -----------------------------------------------------------------------------
// gpu dispatch
// -----------------------------------------------------------------------------

std::vector<f32v3> evaluate_blackbody(std::vector<f32> const & temperatures) {
	u32 const count = (u32)temperatures.size();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "blackbody_evaluate.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto inBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = inBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(temperatures.data()),
			temperatures.size() * sizeof(f32)
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

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("blackbody: gpu matches cpu f64 reference across the fire/glow range") {
	std::vector<f32> temperatures;
	for (f32 t = 800.0f; t <= 15000.0f; t += 137.0f) {
		temperatures.push_back(t);
	}
	auto const out = evaluate_blackbody(temperatures);
	REQUIRE(out.size() == temperatures.size());
	for (u32 i = 0; i < temperatures.size(); ++i) {
		CAPTURE(temperatures[i]);
		RefRgb const ref = ref_blackbody_radiance((f64)temperatures[i]);
		CHECK(std::isfinite(out[i].x));
		CHECK(std::isfinite(out[i].y));
		CHECK(std::isfinite(out[i].z));
		// f64 reference vs gpu f32: loose-ish relative tolerance, magnitudes
		// span many orders across this range (stefan-boltzmann T^4)
		CHECK(out[i].x == doctest::Approx(ref.r).epsilon(0.01).scale(1e-6));
		CHECK(out[i].y == doctest::Approx(ref.g).epsilon(0.01).scale(1e-6));
		CHECK(out[i].z == doctest::Approx(ref.b).epsilon(0.01).scale(1e-6));
	}
}

TEST_CASE("blackbody: never negative, never non-finite, across a wide sweep") {
	std::vector<f32> temperatures;
	for (f32 t = 1.0f; t <= 40000.0f; t += 53.0f) {
		temperatures.push_back(t);
	}
	auto const out = evaluate_blackbody(temperatures);
	u32 nonFinite = 0u;
	u32 negative = 0u;
	for (auto const & c : out) {
		if (!std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.z)) {
			nonFinite++;
		}
		if (c.x < 0.0f || c.y < 0.0f || c.z < 0.0f) {
			negative++;
		}
	}
	CAPTURE(nonFinite);
	CAPTURE(negative);
	CHECK(nonFinite == 0u);
	CHECK(negative == 0u);
}

TEST_CASE("blackbody: brightness grows monotonically with temperature (stefan-boltzmann)") {
	// luminance (cie Y-ish proxy: plain rgb sum) must strictly increase with
	// temperature -- radiated power scales as T^4, unconditionally, for a
	// blackbody at any temperature
	std::vector<f32> const temperatures = {
		500.0f, 1000.0f, 1500.0f, 2000.0f, 3000.0f, 4500.0f,
		6500.0f, 9000.0f, 12000.0f, 20000.0f, 35000.0f,
	};
	auto const out = evaluate_blackbody(temperatures);
	for (u32 i = 1; i < out.size(); ++i) {
		f32 const prevSum = out[i-1].x + out[i-1].y + out[i-1].z;
		f32 const curSum = out[i].x + out[i].y + out[i].z;
		CAPTURE(temperatures[i-1]);
		CAPTURE(temperatures[i]);
		CHECK(curSum > prevSum);
	}
}

TEST_CASE("blackbody: color shifts red-to-blue-white with increasing temperature (wien's law)") {
	// low temperature: red channel should dominate blue (embers, candlelight).
	// high temperature: blue channel should be comparable to or exceed red
	// (blue-white hot). this is the qualitative wien's displacement law
	// signature -- peak emission wavelength shortens as temperature rises
	auto const low = evaluate_blackbody({ 1500.0f })[0];
	auto const high = evaluate_blackbody({ 15000.0f })[0];
	CAPTURE(low.x); CAPTURE(low.y); CAPTURE(low.z);
	CAPTURE(high.x); CAPTURE(high.y); CAPTURE(high.z);
	CHECK(low.x > low.z);
	CHECK(high.z >= high.x);
	// the ratio of blue to red must itself increase with temperature --
	// a direct, unit-independent check of the hue shift (robust to
	// whatever overall brightness scale either sample lands at)
	CHECK((high.z / std::max(high.x, 1e-12f)) > (low.z / std::max(low.x, 1e-12f)));
}

} // TEST_SUITE("[headless]")
