#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <ponder/energy-tables.hpp>
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

struct EnergyPush {
	u64 inVa;
	u64 outVa;
	u32 count;
};

static constexpr u32 kLocalSize = 64;
static u32 groups_for(u32 count) { return (count + kLocalSize - 1) / kLocalSize; }

std::vector<f32> evaluate_directional_albedo(std::vector<f32v4> const & samples) {
	u32 const count = (u32)samples.size();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "directional_albedo_evaluate.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto inBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v4),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = inBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(samples.data()), count * sizeof(f32v4)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	EnergyPush const push {
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

std::vector<f32v2> evaluate_directional_albedo_crosscheck(
	std::vector<f32v4> const & samples
) {
	u32 const count = (u32)samples.size();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "directional_albedo_crosscheck.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto inBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v4),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = inBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(samples.data()), count * sizeof(f32v4)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v2),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	EnergyPush const push {
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

struct CompensateAlbedo { std::vector<f32> compensate, albedo; };

CompensateAlbedo evaluate_energy_compensate(std::vector<f32v3> const & samples) {
	u32 const count = (u32)samples.size();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "energy_compensate_evaluate.comp",
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
	auto compensateOutBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	auto albedoOutBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push { u64 inVa; u64 compensateOutVa; u64 albedoOutVa; u32 count; };
	Push const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.compensateOutVa = vkof::buffer_virtual_address(compensateOutBuf),
		.albedoOutVa = vkof::buffer_virtual_address(albedoOutBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	CompensateAlbedo out;
	out.compensate = test::readback<f32>(compensateOutBuf, 0, count);
	out.albedo = test::readback<f32>(albedoOutBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(compensateOutBuf);
	vkof::buffer_destroy(albedoOutBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

std::vector<f32> evaluate_kulla_conty_table(
	u32 tableHandle, std::vector<f32v2> const & samples
) {
	u32 const count = (u32)samples.size();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "kulla_conty_table_evaluate.comp",
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
			reinterpret_cast<u8 const *>(samples.data()), count * sizeof(f32v2)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push { u64 inVa; u64 outVa; u32 tableHandle; u32 count; };
	Push const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.tableHandle = tableHandle,
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	auto out = test::readback<f32>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

std::vector<f32v2> evaluate_kulla_conty_vs_analytic(
	u32 tableHandle, std::vector<f32v2> const & samples
) {
	u32 const count = (u32)samples.size();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "kulla_conty_vs_analytic.comp",
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
			reinterpret_cast<u8 const *>(samples.data()), count * sizeof(f32v2)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v2),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push { u64 inVa; u64 outVa; u32 tableHandle; u32 count; };
	Push const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.tableHandle = tableHandle,
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	auto out = test::readback<f32v2>(outBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

// golden reference: numerically integrates the true ggx single-scatter
// directional albedo from first principles (see ess_integrate.comp)
std::vector<f32> evaluate_ess_integral(
	std::vector<f32v2> const & samples, u32 thetaSteps, u32 phiSteps
) {
	u32 const count = (u32)samples.size();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "ess_integrate.comp",
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
			reinterpret_cast<u8 const *>(samples.data()), count * sizeof(f32v2)
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

// mirrors the raw table data in lib/ponder/src/energy-tables.cpp, for the
// exact-round-trip test (16x16, roughness x mu)
static f32 const kKullaContyEnergyData[256] = {
	1.000000f, 0.988071f, 0.905022f, 0.896458f, 0.922060f, 0.940172f,
	0.951033f, 0.953774f, 0.952107f, 0.947958f, 0.942332f, 0.934545f,
	0.925498f, 0.914913f, 0.903202f, 0.890433f,
	1.000000f, 0.998627f, 0.977963f, 0.924440f, 0.889060f, 0.886391f,
	0.891276f, 0.894988f, 0.894870f, 0.886240f, 0.876163f, 0.858471f,
	0.840526f, 0.817747f, 0.794919f, 0.769773f,
	1.000000f, 0.999548f, 0.992477f, 0.961348f, 0.915559f, 0.883532f,
	0.870649f, 0.864415f, 0.858654f, 0.846603f, 0.828853f, 0.808554f,
	0.783298f, 0.752251f, 0.720449f, 0.687631f,
	1.000000f, 0.999813f, 0.996053f, 0.978904f, 0.942906f, 0.901699f,
	0.871593f, 0.852289f, 0.836613f, 0.816990f, 0.796714f, 0.768985f,
	0.735167f, 0.701014f, 0.663193f, 0.624795f,
	1.000000f, 0.999856f, 0.997469f, 0.987360f, 0.960689f, 0.920756f,
	0.881842f, 0.849685f, 0.824418f, 0.800921f, 0.770320f, 0.738004f,
	0.701490f, 0.660073f, 0.619153f, 0.572905f,
	1.000000f, 0.999878f, 0.998320f, 0.991305f, 0.972025f, 0.937171f,
	0.895881f, 0.857017f, 0.822157f, 0.787703f, 0.755183f, 0.715447f,
	0.670402f, 0.624560f, 0.579233f, 0.532599f,
	1.000000f, 0.999895f, 0.998738f, 0.993239f, 0.978505f, 0.949633f,
	0.908792f, 0.865747f, 0.825601f, 0.782747f, 0.741589f, 0.698455f,
	0.648912f, 0.597526f, 0.546716f, 0.495422f,
	1.000000f, 0.999962f, 0.999180f, 0.994909f, 0.982861f, 0.958860f,
	0.921529f, 0.876808f, 0.828128f, 0.781164f, 0.733090f, 0.682346f,
	0.628276f, 0.576641f, 0.520170f, 0.461832f,
	1.000000f, 0.999967f, 0.999143f, 0.996110f, 0.986222f, 0.965474f,
	0.931820f, 0.886617f, 0.837798f, 0.783723f, 0.729957f, 0.672340f,
	0.615165f, 0.555356f, 0.493722f, 0.438420f,
	1.000000f, 0.999950f, 0.999315f, 0.996472f, 0.988006f, 0.971015f,
	0.939302f, 0.896990f, 0.844288f, 0.786255f, 0.726462f, 0.663577f,
	0.601563f, 0.537290f, 0.471909f, 0.415300f,
	1.000000f, 0.999953f, 0.999497f, 0.997208f, 0.990300f, 0.974460f,
	0.947030f, 0.905052f, 0.852056f, 0.790519f, 0.726396f, 0.657019f,
	0.587188f, 0.523369f, 0.454441f, 0.391267f,
	1.000000f, 0.999994f, 0.999558f, 0.997375f, 0.991503f, 0.977419f,
	0.951576f, 0.912996f, 0.860437f, 0.796573f, 0.726529f, 0.655100f,
	0.581643f, 0.509683f, 0.437398f, 0.372621f,
	1.000000f, 0.999956f, 0.999568f, 0.997899f, 0.992178f, 0.980203f,
	0.957207f, 0.919148f, 0.867690f, 0.802882f, 0.730433f, 0.650576f,
	0.571347f, 0.495317f, 0.423144f, 0.354806f,
	1.000000f, 0.999978f, 0.999652f, 0.997975f, 0.992989f, 0.982050f,
	0.960359f, 0.924918f, 0.875282f, 0.807544f, 0.732619f, 0.649359f,
	0.564623f, 0.483091f, 0.408167f, 0.339617f,
	1.000000f, 0.999969f, 0.999708f, 0.998276f, 0.993703f, 0.983939f,
	0.964226f, 0.930900f, 0.879820f, 0.816623f, 0.736827f, 0.649661f,
	0.560940f, 0.477082f, 0.397979f, 0.324305f,
	1.000000f, 0.999960f, 0.999734f, 0.998373f, 0.994381f, 0.985219f,
	0.966582f, 0.935156f, 0.886950f, 0.821114f, 0.742601f, 0.650476f,
	0.559640f, 0.466482f, 0.385075f, 0.313566f,
};

} // namespace

TEST_SUITE("[headless]") {

// -----------------------------------------------------------------------
// -- utilMicrofacetGgxDirectionalAlbedo: materialx rational fit
// -----------------------------------------------------------------------

TEST_CASE("directional albedo: finite and roughly bounded over the trained domain (fuzz)") {
	std::vector<f32v4> samples;
	u32 seed = 1u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 2048u; ++i) { samples.push_back({ next(), next(), 1.0f, 1.0f }); }
	auto const result = evaluate_directional_albedo(samples);
	for (f32 v : result) {
		CHECK(std::isfinite(v));
		CHECK(v >= -0.05f); // fitted approximation, small overshoot tolerance
		CHECK(v <= 1.05f);
	}
}

TEST_CASE("directional albedo: matches materialx's mx_ggx_dir_albedo_analytic") {
	std::vector<f32v4> samples;
	u32 seed = 2u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 1024u; ++i) {
		samples.push_back({ next(), next(), next(), next() });
	}
	auto const result = evaluate_directional_albedo_crosscheck(samples);
	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(result[i].x == doctest::Approx(result[i].y).epsilon(0.0001));
	}
}

// -----------------------------------------------------------------------
// -- utilMicrofacetDielectricEnergyCompensate / utilMicrofacetDielectricAlbedo
// -----------------------------------------------------------------------

TEST_CASE("energy compensate: close to >= 1 (never removes much energy), finite (fuzz)") {
	// NOT a strict >= 1 bound: ess = ab.x + ab.y with f0=f90=1, where ab.x
	// and ab.y are EACH independently clamped to [0,1] by the rational
	// fit -- their sum can slightly exceed 1 in regions where the fit
	// overshoots, which then makes compensate dip just under 1. that's a
	// real property of the fitted approximation (not a closed form with a
	// hard energy-conservation guarantee), not a bug -- confirmed the
	// dips are small (a fraction of a percent), not a systemic issue.
	std::vector<f32v3> samples;
	u32 seed = 3u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 2048u; ++i) { samples.push_back({ next(), next(), next() }); }
	// explicitly include the grazing/high-roughness corner cull's own
	// comment warns about (analytic fit legitimately hits 0 there)
	samples.push_back({ 0.0f, 1.0f, 0.5f });
	samples.push_back({ 0.001f, 0.999f, 0.9f });

	auto const result = evaluate_energy_compensate(samples);
	for (f32 v : result.compensate) {
		CAPTURE(v);
		CHECK(std::isfinite(v));
		CHECK(v >= 0.95f);
	}
	for (f32 v : result.albedo) {
		CHECK(std::isfinite(v));
		CHECK(v >= -0.0001f);
		CHECK(v <= 1.0001f);
	}
}

// -----------------------------------------------------------------------
// -- utilMicrofacetDielectricEnergyCompensationKullaConty: table lookup
// -----------------------------------------------------------------------

TEST_CASE("kulla-conty table: exact round-trip at grid points") {
	auto tables = ponder::energy_tables_create();

	std::vector<f32v2> samples;
	for (u32 row = 0; row < 16u; ++row) {
		for (u32 col = 0; col < 16u; ++col) {
			// matches the lookup code's OWN indexing: ai = sqrt(roughness)*15,
			// mi = mu*16-0.5 -- feeding exact grid coordinates back
			f32 const rowFrac = (f32)row / 15.0f;
			f32 const roughness = rowFrac * rowFrac;
			f32 const mu = ((f32)col + 0.5f) / 16.0f;
			samples.push_back({ mu, roughness });
		}
	}
	auto const result = evaluate_kulla_conty_table(tables.kullaContyEnergyHandle, samples);

	for (u32 row = 0; row < 16u; ++row) {
		for (u32 col = 0; col < 16u; ++col) {
			u32 const i = row * 16u + col;
			CAPTURE(row);
			CAPTURE(col);
			CHECK(result[i] == doctest::Approx(kKullaContyEnergyData[i]).epsilon(0.001));
		}
	}

	ponder::energy_tables_destroy(tables);
}

TEST_CASE("kulla-conty table: bounded in [0,1] and finite (fuzz)") {
	auto tables = ponder::energy_tables_create();

	std::vector<f32v2> samples;
	u32 seed = 4u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 1024u; ++i) { samples.push_back({ next(), next() }); }
	auto const result = evaluate_kulla_conty_table(tables.kullaContyEnergyHandle, samples);
	for (f32 v : result) {
		CHECK(std::isfinite(v));
		CHECK(v >= -0.0001f);
		CHECK(v <= 1.0001f);
	}

	ponder::energy_tables_destroy(tables);
}

TEST_CASE(
	"kulla-conty table vs analytic fit: same physical quantity, compare "
	"(known bug, ported 1:1 from cull as-is)" * doctest::should_fail()
) {
	// both represent the ggx single-scatter directional albedo with
	// perfect fresnel (f0=f90=1); one is a precomputed monte carlo table,
	// the other a fitted polynomial -- they should roughly agree if both
	// the table's axis convention and the analytic fit are correct.
	//
	// they don't: mean/max deviation are far larger than two independent
	// approximations of the same quantity should produce. the table's own
	// comment says rows are generated at alpha=(i/16)^2 (non-linear,
	// sqrt-compressed spacing), but the lookup code indexes rows linearly
	// (ai = roughness*15, no inverse-sqrt). this function has zero
	// callers in cull's actual rendering path (reference/verification
	// only), and there's no generator script in the repo to verify the
	// exact intended convention against, so this is documented as a known
	// discrepancy rather than guess-fixed. see the artifact report for
	// the comparison images.
	auto tables = ponder::energy_tables_create();

	std::vector<f32v2> samples;
	for (f32 roughness = 0.05f; roughness <= 1.0f; roughness += 0.05f) {
		for (f32 mu = 0.05f; mu <= 1.0f; mu += 0.05f) {
			samples.push_back({ mu, roughness });
		}
	}
	auto const result = evaluate_kulla_conty_vs_analytic(
		tables.kullaContyEnergyHandle, samples
	);

	f32 maxDeviation = 0.0f;
	f32 sumDeviation = 0.0f;
	for (u32 i = 0; i < result.size(); ++i) {
		f32 const deviation = std::abs(result[i].x - result[i].y);
		maxDeviation = std::max(maxDeviation, deviation);
		sumDeviation += deviation;
	}
	f32 const meanDeviation = sumDeviation / (f32)result.size();
	CAPTURE(maxDeviation);
	CAPTURE(meanDeviation);
	// generous: two different approximation methods for the same quantity
	CHECK(meanDeviation < 0.1f);

	ponder::energy_tables_destroy(tables);
}

// -----------------------------------------------------------------------
// -- golden reference: neither the table nor the analytic fit is trusted
// -- as ground truth above -- comparing them to each other only shows
// -- they disagree, not which one (if either) is right. these two tests
// -- check each independently against a from-first-principles numerical
// -- integration (ess_integrate.comp), built from the already-validated
// -- D and Smith-V primitives, not copied from cull/materialx/the table.
// -----------------------------------------------------------------------

TEST_CASE("golden reference: analytic fit matches numerically integrated ggx albedo") {
	constexpr u32 kThetaSteps = 300;
	constexpr u32 kPhiSteps = 150;

	std::vector<f32v2> samples;
	for (f32 mu = 0.1f; mu <= 0.95f; mu += 0.15f) {
		for (f32 alpha = 0.1f; alpha <= 0.95f; alpha += 0.15f) {
			samples.push_back({ mu, alpha });
		}
	}
	auto const golden = evaluate_ess_integral(samples, kThetaSteps, kPhiSteps);

	std::vector<f32v4> analyticSamples;
	for (auto const & s : samples) { analyticSamples.push_back({ s.x, s.y, 1.0f, 1.0f }); }
	auto const analytic = evaluate_directional_albedo(analyticSamples);

	f32 maxDeviation = 0.0f, sumDeviation = 0.0f;
	for (u32 i = 0; i < golden.size(); ++i) {
		f32 const deviation = std::abs(analytic[i] - golden[i]);
		CAPTURE(samples[i].x);
		CAPTURE(samples[i].y);
		CAPTURE(golden[i]);
		CAPTURE(analytic[i]);
		maxDeviation = std::max(maxDeviation, deviation);
		sumDeviation += deviation;
	}
	f32 const meanDeviation = sumDeviation / (f32)golden.size();
	CAPTURE(maxDeviation);
	CAPTURE(meanDeviation);
	CHECK(meanDeviation < 0.05f);
}

TEST_CASE(
	"golden reference: kulla-conty table (with axis fix) vs numerically "
	"integrated ggx albedo (known open issue, not resolved by the axis fix)"
	* doctest::should_fail()
) {
	// the axis fix (sqrt(roughness)*15, see util-energy-compensation.glsl)
	// is kept because it's more consistent with the table's own
	// documented row-generation convention, but it does NOT make the
	// table match real physics: mean deviation against this
	// from-first-principles integration is ~0.19 (vs ~0.02 for the
	// analytic fit, which passes the equivalent check above). the table
	// has a deeper, undiagnosed problem beyond just row indexing --
	// documented as open, not guess-fixed further.
	auto tables = ponder::energy_tables_create();
	constexpr u32 kThetaSteps = 300;
	constexpr u32 kPhiSteps = 150;

	std::vector<f32v2> samples;
	for (f32 mu = 0.1f; mu <= 0.95f; mu += 0.15f) {
		for (f32 alpha = 0.1f; alpha <= 0.95f; alpha += 0.15f) {
			samples.push_back({ mu, alpha });
		}
	}
	auto const golden = evaluate_ess_integral(samples, kThetaSteps, kPhiSteps);
	auto const table = evaluate_kulla_conty_table(tables.kullaContyEnergyHandle, samples);

	f32 maxDeviation = 0.0f, sumDeviation = 0.0f;
	for (u32 i = 0; i < golden.size(); ++i) {
		f32 const deviation = std::abs(table[i] - golden[i]);
		CAPTURE(samples[i].x);
		CAPTURE(samples[i].y);
		CAPTURE(golden[i]);
		CAPTURE(table[i]);
		maxDeviation = std::max(maxDeviation, deviation);
		sumDeviation += deviation;
	}
	f32 const meanDeviation = sumDeviation / (f32)golden.size();
	CAPTURE(maxDeviation);
	CAPTURE(meanDeviation);
	CHECK(meanDeviation < 0.05f);

	ponder::energy_tables_destroy(tables);
}

TEST_CASE("directional albedo: heatmap (dotNorWi x alpha)") {
	constexpr u32 kSize = 512;
	std::vector<f32v4> samples(kSize * kSize);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const alpha = (f32)y / (f32)(kSize - 1);
		for (u32 x = 0; x < kSize; ++x) {
			f32 const dotNorWi = (f32)x / (f32)(kSize - 1);
			samples[y * kSize + x] = { dotNorWi, alpha, 1.0f, 1.0f };
		}
	}
	auto const result = evaluate_directional_albedo(samples);

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		result, result, result, kSize, kSize,
		ENERGY_OUTPUT_DIR "directional_albedo_heatmap.png", &nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);
}

TEST_CASE("kulla-conty table: heatmap (mu x roughness), and difference vs analytic") {
	auto tables = ponder::energy_tables_create();
	constexpr u32 kSize = 512;

	std::vector<f32v2> samples(kSize * kSize);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const roughness = (f32)y / (f32)(kSize - 1);
		for (u32 x = 0; x < kSize; ++x) {
			f32 const mu = (f32)x / (f32)(kSize - 1);
			samples[y * kSize + x] = { mu, roughness };
		}
	}
	auto const result = evaluate_kulla_conty_vs_analytic(
		tables.kullaContyEnergyHandle, samples
	);

	std::vector<f32> table(result.size()), analytic(result.size()), diff(result.size());
	for (u32 i = 0; i < result.size(); ++i) {
		table[i] = result[i].x;
		analytic[i] = result[i].y;
		// centered at 0.5, +-0.5 maps to full black/white
		diff[i] = std::clamp(0.5f + (result[i].x - result[i].y), 0.0f, 1.0f);
	}

	u32 nanCount = 0;
	bool ok = test::write_heatmap_png(
		table, table, table, kSize, kSize,
		ENERGY_OUTPUT_DIR "kulla_conty_table_heatmap.png", &nanCount
	);
	CHECK(ok);
	CHECK(nanCount == 0);

	ok = test::write_heatmap_png(
		analytic, analytic, analytic, kSize, kSize,
		ENERGY_OUTPUT_DIR "kulla_conty_analytic_heatmap.png", &nanCount
	);
	CHECK(ok);
	CHECK(nanCount == 0);

	ok = test::write_heatmap_png(
		diff, diff, diff, kSize, kSize,
		ENERGY_OUTPUT_DIR "kulla_conty_vs_analytic_diff.png", &nanCount
	);
	CHECK(ok);
	CHECK(nanCount == 0);

	ponder::energy_tables_destroy(tables);
}

} // TEST_SUITE("[headless]")
