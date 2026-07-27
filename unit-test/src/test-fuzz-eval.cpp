#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <ponder/zeltner-tables.hpp>
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// verification for the remainder of the ported zeltner ltc fuzz (sheen)
// lobe: utilZeltnerFuzzLookup (the 32x32 fit table + bilinear lookup) and
// openPbrFuzzEvaluateF (the layered eval, built on the already-verified
// openPbrFuzzSampleWo/openPbrFuzzPdf pair -- see test-fuzz-sampling.cpp).
// this milestone also lands the C++ table generator (zeltner-tables.hpp/
// cpp, ported 1:1 from cull's material-tables.cpp) that utilZeltnerFuzzLookup
// needed but had never been wired up in ponder.
// ---------------------------------------------------------------------------

namespace {

static constexpr u32 kLocalSize = 64;
static u32 groups_for(u32 count) { return (count + kLocalSize - 1) / kLocalSize; }

// -----------------------------------------------------------------------------
// zeltner_lookup_evaluate.comp / zeltner_lookup_crosscheck.comp runners
// -----------------------------------------------------------------------------

std::vector<f32> run_lookup(
	u32 tableHandle, std::vector<f32> const & muRoughFlat
) {
	u32 const count = (u32)muRoughFlat.size() / 2u;
	REQUIRE(muRoughFlat.size() == (size_t)count * 2u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "zeltner_lookup_evaluate.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto inBuf = vkof::buffer_create({
		.byteCount = muRoughFlat.size() * sizeof(f32),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = inBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(muRoughFlat.data()),
			muRoughFlat.size() * sizeof(f32)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = (u64)count * 3u * sizeof(f32),
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

	auto out = test::readback<f32>(outBuf, 0, count * 3u);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

std::vector<f32> run_lookup_crosscheck(
	u32 tableHandle, std::vector<f32> const & muRoughFlat
) {
	u32 const count = (u32)muRoughFlat.size() / 2u;
	REQUIRE(muRoughFlat.size() == (size_t)count * 2u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "zeltner_lookup_crosscheck.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto inBuf = vkof::buffer_create({
		.byteCount = muRoughFlat.size() * sizeof(f32),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = inBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(muRoughFlat.data()),
			muRoughFlat.size() * sizeof(f32)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = (u64)count * 6u * sizeof(f32),
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

	auto out = test::readback<f32>(outBuf, 0, count * 6u);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

// -----------------------------------------------------------------------------
// fuzz_evaluate_f_evaluate.comp runner
// -----------------------------------------------------------------------------

void push_eval_sample(
	std::vector<f32> & flat, f32v3 nor, f32v3 wi, f32v3 wo,
	f32 fuzzWeight, f32v3 fuzzColor, f32 fuzzRoughness, f32v3 coatedBase
) {
	flat.push_back(nor.x); flat.push_back(nor.y); flat.push_back(nor.z);
	flat.push_back(wi.x); flat.push_back(wi.y); flat.push_back(wi.z);
	flat.push_back(wo.x); flat.push_back(wo.y); flat.push_back(wo.z);
	flat.push_back(fuzzWeight);
	flat.push_back(fuzzColor.x); flat.push_back(fuzzColor.y); flat.push_back(fuzzColor.z);
	flat.push_back(fuzzRoughness);
	flat.push_back(coatedBase.x); flat.push_back(coatedBase.y); flat.push_back(coatedBase.z);
}

std::vector<f32> run_eval(u32 tableHandle, std::vector<f32> const & flat) {
	u32 const count = (u32)flat.size() / 17u;
	REQUIRE(flat.size() == (size_t)count * 17u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fuzz_evaluate_f_evaluate.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto inBuf = vkof::buffer_create({
		.byteCount = flat.size() * sizeof(f32),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = inBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(flat.data()), flat.size() * sizeof(f32)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = (u64)count * 3u * sizeof(f32),
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

	auto out = test::readback<f32>(outBuf, 0, count * 3u);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

// reuse fuzz_pdf_evaluate.comp from the sampling milestone: nor.xyz, wi.xyz,
// wo.xyz, fuzzA, fuzzB -> pdf
std::vector<f32> run_pdf(std::vector<f32> const & flat) {
	u32 const count = (u32)flat.size() / 11u;
	REQUIRE(flat.size() == (size_t)count * 11u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fuzz_pdf_evaluate.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto inBuf = vkof::buffer_create({
		.byteCount = flat.size() * sizeof(f32),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = inBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(flat.data()), flat.size() * sizeof(f32)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = (u64)count * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push { u64 inVa; u64 outVa; u32 count; };
	Push const push {
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

// -----------------------------------------------------------------------------
// fuzz_furnace_integrate.comp runner
// -----------------------------------------------------------------------------

void push_furnace_sample(
	std::vector<f32> & flat, f32v3 wi,
	f32 fuzzWeight, f32v3 fuzzColor, f32 fuzzRoughness, f32v3 coatedBase
) {
	flat.push_back(wi.x); flat.push_back(wi.y); flat.push_back(wi.z);
	flat.push_back(fuzzWeight);
	flat.push_back(fuzzColor.x); flat.push_back(fuzzColor.y); flat.push_back(fuzzColor.z);
	flat.push_back(fuzzRoughness);
	flat.push_back(coatedBase.x); flat.push_back(coatedBase.y); flat.push_back(coatedBase.z);
}

std::vector<f32> run_furnace(
	u32 tableHandle, std::vector<f32> const & flat, u32 thetaSteps, u32 phiSteps
) {
	u32 const count = (u32)flat.size() / 11u;
	REQUIRE(flat.size() == (size_t)count * 11u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fuzz_furnace_integrate.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto inBuf = vkof::buffer_create({
		.byteCount = flat.size() * sizeof(f32),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = inBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(flat.data()), flat.size() * sizeof(f32)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = (u64)count * 3u * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push {
		u64 inVa; u64 outVa; u32 tableHandle; u32 thetaSteps; u32 phiSteps; u32 count;
	};
	Push const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.tableHandle = tableHandle,
		.thetaSteps = thetaSteps,
		.phiSteps = phiSteps,
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	auto out = test::readback<f32>(outBuf, 0, count * 3u);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("zeltner lut: spot-check known table entries") {
	// texel-center (mu, roughness) reduce the bilinear fetch to an exact
	// single-texel read. mu axis is (i+0.5)/32 per utilZeltnerFuzzLookup's
	// own comment, so mu = 0.5/32 hits mu bin 0 exactly. the roughness
	// axis is NOT the "(i/32)^2" the same comment describes -- that
	// documents the offline fit's sampling density (denser near smooth),
	// not the runtime lookup, which maps a query roughness to a row via
	// the plain linear `ai = roughness * 31`; so roughness = r/31 is what
	// hits row r exactly (confirmed by the materialx cross-check test
	// below tracking well under the *linear* mapping, and empirically:
	// (r/32)^2 landed off by a full row for r=1). row 31 is a special
	// case -- the clamp `ai <= 30.999` means it can only ever be reached
	// as a ~99.9/0.1 blend with row 30, never a pure fetch, hence the
	// looser tolerance there.
	auto tables = ponder::zeltner_tables_create();

	std::vector<f32> flat = {
		0.5f / 32.0f, 0.0f,
		0.5f / 32.0f, 1.0f / 31.0f,
		31.5f / 32.0f, 1.0f,
	};
	auto const out = run_lookup(tables.zeltnerLtcParamHandle, flat);

	CHECK(out[0] == doctest::Approx(0.01415).epsilon(1e-3));
	CHECK(out[1] == doctest::Approx(0.00060).epsilon(2e-2));
	CHECK(out[2] == doctest::Approx(0.00001).epsilon(2e-1));

	CHECK(out[3] == doctest::Approx(0.01941).epsilon(1e-3));
	CHECK(out[4] == doctest::Approx(-0.00232).epsilon(2e-2));
	CHECK(out[5] == doctest::Approx(0.05839).epsilon(1e-3));

	CHECK(out[6] == doctest::Approx(0.87958).epsilon(5e-3));
	CHECK(out[7] == doctest::Approx(0.00003).epsilon(2e-1));
	CHECK(out[8] == doctest::Approx(0.34187).epsilon(5e-3));

	ponder::zeltner_tables_destroy(tables);
}

TEST_CASE("zeltner lut: NaN/bounds sweep + R-channel heatmap") {
	auto tables = ponder::zeltner_tables_create();

	constexpr u32 kSize = 256u;
	std::vector<f32> flat;
	flat.reserve((size_t)kSize * kSize * 2u);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const roughness = (f32)y / (f32)(kSize - 1);
		for (u32 x = 0; x < kSize; ++x) {
			f32 const mu = (f32)x / (f32)(kSize - 1);
			flat.push_back(mu);
			flat.push_back(roughness);
		}
	}
	auto const out = run_lookup(tables.zeltnerLtcParamHandle, flat);

	std::vector<f32> img(kSize * kSize);
	u32 nonFinite = 0u;
	for (u32 i = 0; i < kSize * kSize; ++i) {
		f32 const a = out[i*3+0], b = out[i*3+1], r = out[i*3+2];
		bool const finite = std::isfinite(a) && std::isfinite(b) && std::isfinite(r);
		if (!finite) { nonFinite++; img[i] = std::nanf(""); continue; }
		img[i] = r;
	}
	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		img, img, img, kSize, kSize,
		FUZZ_OUTPUT_DIR "zeltner_lut_r_channel.png", &nanCount
	);
	CHECK(ok);
	CHECK(nonFinite == 0u);

	ponder::zeltner_tables_destroy(tables);
}

TEST_CASE("zeltner lut vs materialx rational fit: cross-check away from the sparse near-mirror band") {
	// the tabulated fit is a 32x32 discretization of the same zeltner et
	// al. 2022 data materialx's rational functions fit in closed form --
	// two independent fits of the same underlying numbers, not the same
	// formula, so exact agreement isn't expected. the smoothest rows
	// (roughness -> 0) are almost entirely zero-padded in the table (see
	// the spot-check test's index-0 row), a genuine sparse-fit artifact
	// rather than something materialx's smooth analytic curve will track,
	// so this excludes roughness < 0.15.
	auto tables = ponder::zeltner_tables_create();

	constexpr u32 kSize = 96u;
	std::vector<f32> flat;
	std::vector<f32> roughs;
	for (u32 y = 0; y < kSize; ++y) {
		f32 const roughness = 0.15f + (1.0f - 0.15f) * (f32)y / (f32)(kSize - 1);
		for (u32 x = 0; x < kSize; ++x) {
			f32 const mu = 0.02f + 0.98f * (f32)x / (f32)(kSize - 1);
			flat.push_back(mu);
			flat.push_back(roughness);
			roughs.push_back(roughness);
		}
	}
	auto const out = run_lookup_crosscheck(tables.zeltnerLtcParamHandle, flat);

	f64 worstA = 0.0, worstB = 0.0, worstR = 0.0;
	u32 nonFinite = 0u;
	for (u32 i = 0; i < roughs.size(); ++i) {
		f32 const tableA = out[i*6+0], tableB = out[i*6+1], tableR = out[i*6+2];
		f32 const refA = out[i*6+3], refB = out[i*6+4], refR = out[i*6+5];
		bool const finite = (
			std::isfinite(tableA) && std::isfinite(tableB) && std::isfinite(tableR)
			&& std::isfinite(refA) && std::isfinite(refB) && std::isfinite(refR)
		);
		CAPTURE(i);
		CHECK(finite);
		if (!finite) { nonFinite++; continue; }
		worstA = std::max(worstA, (f64)std::fabs(tableA - refA));
		worstB = std::max(worstB, (f64)std::fabs(tableB - refB));
		worstR = std::max(worstR, (f64)std::fabs(tableR - refR));
	}
	MESSAGE(
		"table vs materialx worst abs deviation: a=", worstA,
		" b=", worstB, " R=", worstR
	);
	CHECK(nonFinite == 0u);
	// generous: two independent fits of the same data, not the same
	// formula. tightened from an initial empirical run rather than
	// guessed up front.
	CHECK(worstA < 0.25);
	CHECK(worstB < 0.25);
	CHECK(worstR < 0.25);

	ponder::zeltner_tables_destroy(tables);
}

TEST_CASE("fuzz eval: fuzzWeight = 0 reduces to exactly coatedBase") {
	auto tables = ponder::zeltner_tables_create();

	std::vector<f32> flat;
	std::vector<f32v3> bases;
	for (u32 i = 0; i < 64u; ++i) {
		f32v3 const coatedBase { 0.1f * (f32)i, 0.02f * (f32)i, 0.3f };
		bases.push_back(coatedBase);
		push_eval_sample(
			flat, { 0.0f, 0.0f, 1.0f }, { 0.6f, 0.0f, 0.8f }, { 0.3f, 0.1f, 0.947f },
			0.0f, { 1.0f, 1.0f, 1.0f }, 0.5f, coatedBase
		);
	}
	auto const out = run_eval(tables.zeltnerLtcParamHandle, flat);
	for (u32 i = 0; i < bases.size(); ++i) {
		CAPTURE(i);
		CHECK(out[i*3+0] == doctest::Approx(bases[i].x).epsilon(1e-6));
		CHECK(out[i*3+1] == doctest::Approx(bases[i].y).epsilon(1e-6));
		CHECK(out[i*3+2] == doctest::Approx(bases[i].z).epsilon(1e-6));
	}

	ponder::zeltner_tables_destroy(tables);
}

TEST_CASE("fuzz eval: coatedBase = 0, fuzzWeight = 1 matches openPbrFuzzPdf exactly") {
	// eval's own doc comment: f * cos(theta_o) = C_fuzz * R * D(wo), and
	// openPbrFuzzPdf(...) *is* D(wo). so with coatedBase=0 and
	// fuzzWeight=1, openPbrFuzzEvaluateF(...) * dot(nor,wo) must equal
	// fuzzColor * eFuzz(mu_i,roughness) * openPbrFuzzPdf(nor,wi,wo,a,b) at
	// the SAME table-derived (a,b,eFuzz) -- an exact algebraic identity
	// tying the new eval verification directly back to the pdf already
	// verified in test-fuzz-sampling.cpp, independent of materialx.
	auto tables = ponder::zeltner_tables_create();

	struct Case { f32v3 wi; f32v3 wo; f32 fuzzRoughness; f32v3 fuzzColor; };
	std::vector<Case> cases;
	std::vector<f32> const cosThetaIs = { 1.0f, 0.7f, 0.3f, 0.05f };
	std::vector<f32> const roughnesses = { 0.2f, 0.5f, 0.9f };
	for (f32 const ctI : cosThetaIs) {
		f32 const stI = std::sqrt(std::fmax(0.0f, 1.0f - ctI * ctI));
		for (f32 const roughness : roughnesses) {
			cases.push_back({
				{ stI, 0.0f, ctI }, { 0.4f, 0.5f, 0.768f },
				roughness, { 1.0f, 0.6f, 0.2f }
			});
		}
	}

	std::vector<f32> evalFlat, lookupFlat, pdfFlat;
	for (auto const & c : cases) {
		push_eval_sample(
			evalFlat, { 0.0f, 0.0f, 1.0f }, c.wi, c.wo,
			1.0f, c.fuzzColor, c.fuzzRoughness, { 0.0f, 0.0f, 0.0f }
		);
		lookupFlat.push_back(c.wi.z);
		lookupFlat.push_back(c.fuzzRoughness);
	}
	auto const evalOut = run_eval(tables.zeltnerLtcParamHandle, evalFlat);
	auto const lookupOut = run_lookup(tables.zeltnerLtcParamHandle, lookupFlat);
	for (u32 i = 0; i < cases.size(); ++i) {
		f32v3 const wo = cases[i].wo;
		f32 const a = lookupOut[i*3+0], b = lookupOut[i*3+1], r = lookupOut[i*3+2];
		pdfFlat.push_back(0.0f); pdfFlat.push_back(0.0f); pdfFlat.push_back(1.0f);
		pdfFlat.push_back(cases[i].wi.x); pdfFlat.push_back(cases[i].wi.y); pdfFlat.push_back(cases[i].wi.z);
		pdfFlat.push_back(wo.x); pdfFlat.push_back(wo.y); pdfFlat.push_back(wo.z);
		pdfFlat.push_back(a); pdfFlat.push_back(b);
	}
	auto const pdfOut = run_pdf(pdfFlat);

	for (u32 i = 0; i < cases.size(); ++i) {
		CAPTURE(i);
		f32 const r = lookupOut[i*3+2];
		f32v3 const wo = cases[i].wo;
		f32v3 const fuzzColor = cases[i].fuzzColor;
		f32 const dotNorWo = wo.z;
		f32v3 const evalScaled {
			evalOut[i*3+0] * dotNorWo,
			evalOut[i*3+1] * dotNorWo,
			evalOut[i*3+2] * dotNorWo,
		};
		f32 const pdfTerm = r * pdfOut[i];
		f32v3 const expected {
			fuzzColor.x * pdfTerm, fuzzColor.y * pdfTerm, fuzzColor.z * pdfTerm
		};
		CHECK(evalScaled.x == doctest::Approx(expected.x).epsilon(1e-3));
		CHECK(evalScaled.y == doctest::Approx(expected.y).epsilon(1e-3));
		CHECK(evalScaled.z == doctest::Approx(expected.z).epsilon(1e-3));
	}

	ponder::zeltner_tables_destroy(tables);
}

TEST_CASE("fuzz eval: NaN sweep over (incidence x fuzzRoughness), heatmap") {
	auto tables = ponder::zeltner_tables_create();

	constexpr u32 kSize = 128u;
	std::vector<f32> flat;
	flat.reserve((size_t)kSize * kSize * 17u);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const roughness = (f32)y / (f32)(kSize - 1);
		for (u32 x = 0; x < kSize; ++x) {
			f32 const ct = -1.0f + 2.0f * (f32)x / (f32)(kSize - 1);
			f32 const st = std::sqrt(std::fmax(0.0f, 1.0f - ct * ct));
			push_eval_sample(
				flat, { 0.0f, 0.0f, 1.0f }, { st, 0.0f, ct }, { 0.6f, 0.0f, 0.8f },
				1.0f, { 1.0f, 1.0f, 1.0f }, roughness, { 0.0f, 0.0f, 0.0f }
			);
		}
	}
	auto const out = run_eval(tables.zeltnerLtcParamHandle, flat);

	std::vector<f32> img(kSize * kSize);
	u32 nonFinite = 0u;
	for (u32 i = 0; i < kSize * kSize; ++i) {
		f32 const v = out[i*3+0];
		if (!std::isfinite(v) || !std::isfinite(out[i*3+1]) || !std::isfinite(out[i*3+2])) {
			nonFinite++; img[i] = std::nanf(""); continue;
		}
		img[i] = v / (v + 1.0f);
	}
	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		img, img, img, kSize, kSize,
		FUZZ_OUTPUT_DIR "fuzz_eval_nan_sweep.png", &nanCount
	);
	CHECK(ok);
	CHECK(nonFinite == 0u);

	ponder::zeltner_tables_destroy(tables);
}

TEST_CASE("fuzz eval furnace: isolated fuzz-only energy matches the table's own R") {
	// coatedBase=0, fuzzWeight=1: the furnace integral of fuzzColor*fFuzz
	// over the wo hemisphere is, by the paper's own construction, the
	// table's R channel (eFuzz) -- the fitted directional albedo of the
	// LTC lobe. this ties the table data to the actual sampled BRDF
	// formula's own integral, not just an isolated table read.
	auto tables = ponder::zeltner_tables_create();

	std::vector<f32> furnaceFlat, lookupFlat;
	std::vector<f32> const cosThetaIs = { 1.0f, 0.8f, 0.5f, 0.2f, 0.05f };
	std::vector<f32> const roughnesses = { 0.2f, 0.5f, 0.8f };
	for (f32 const ct : cosThetaIs) {
		f32 const st = std::sqrt(std::fmax(0.0f, 1.0f - ct * ct));
		for (f32 const roughness : roughnesses) {
			push_furnace_sample(
				furnaceFlat, { st, 0.0f, ct },
				1.0f, { 1.0f, 1.0f, 1.0f }, roughness, { 0.0f, 0.0f, 0.0f }
			);
			lookupFlat.push_back(ct);
			lookupFlat.push_back(roughness);
		}
	}
	auto const furnaceOut = run_furnace(
		tables.zeltnerLtcParamHandle, furnaceFlat, 128u, 256u
	);
	auto const lookupOut = run_lookup(tables.zeltnerLtcParamHandle, lookupFlat);

	f64 worst = 0.0;
	u32 const n = (u32)lookupOut.size() / 3u;
	for (u32 i = 0; i < n; ++i) {
		f32 const measured = furnaceOut[i*3+0];
		f32 const tableR = lookupOut[i*3+2];
		CAPTURE(i);
		CHECK(std::isfinite(measured));
		f64 const d = std::fabs((f64)measured - (f64)tableR);
		worst = std::max(worst, d);
		CHECK(d < 0.05);
	}
	MESSAGE("fuzz furnace vs table R worst |diff|: ", worst);

	ponder::zeltner_tables_destroy(tables);
}

TEST_CASE("fuzz eval furnace: layered energy matches the closed-form combination") {
	// with a constant (wo-independent) coatedBase = k/pi, integral_wo
	// coatedBase * cos(theta_o) dwo = k exactly, so the furnace integral
	// of the FULL layered eval must equal
	//   fuzzWeight * R_fuzz_actual(wi) + (1 - fuzzWeight*eFuzz(wi)) * k
	// where R_fuzz_actual(wi) is measured directly (previous test case),
	// not assumed equal to the table's R -- isolates the layering
	// arithmetic's correctness from whether the table matches its own
	// claimed integral.
	auto tables = ponder::zeltner_tables_create();
	constexpr f32 kPi = 3.14159265358979323846f;

	struct Case { f32v3 wi; f32 fuzzWeight; f32 fuzzRoughness; f32 k; };
	std::vector<Case> cases;
	std::vector<f32> const cosThetaIs = { 1.0f, 0.6f, 0.2f };
	std::vector<f32> const weights = { 0.3f, 0.7f, 1.0f };
	for (f32 const ct : cosThetaIs) {
		for (f32 const w : weights) {
			cases.push_back({ { 0.0f, 0.0f, 0.0f }, w, 0.5f, 0.4f });
			cases.back().wi = {
				std::sqrt(std::fmax(0.0f, 1.0f - ct * ct)), 0.0f, ct
			};
		}
	}

	std::vector<f32> isolatedFlat, layeredFlat, lookupFlat;
	for (auto const & c : cases) {
		push_furnace_sample(
			isolatedFlat, c.wi, 1.0f, { 1.0f, 1.0f, 1.0f }, c.fuzzRoughness,
			{ 0.0f, 0.0f, 0.0f }
		);
		push_furnace_sample(
			layeredFlat, c.wi, c.fuzzWeight, { 1.0f, 1.0f, 1.0f }, c.fuzzRoughness,
			{ c.k / kPi, c.k / kPi, c.k / kPi }
		);
		lookupFlat.push_back(c.wi.z);
		lookupFlat.push_back(c.fuzzRoughness);
	}
	auto const isolatedOut = run_furnace(
		tables.zeltnerLtcParamHandle, isolatedFlat, 128u, 256u
	);
	auto const layeredOut = run_furnace(
		tables.zeltnerLtcParamHandle, layeredFlat, 128u, 256u
	);
	auto const lookupOut = run_lookup(tables.zeltnerLtcParamHandle, lookupFlat);

	f64 worst = 0.0;
	for (u32 i = 0; i < cases.size(); ++i) {
		f32 const rFuzzActual = isolatedOut[i*3+0];
		f32 const eFuzzTable = lookupOut[i*3+2];
		f64 const expected = (
			(f64)cases[i].fuzzWeight * (f64)rFuzzActual
			+ (1.0 - (f64)cases[i].fuzzWeight * (f64)eFuzzTable) * (f64)cases[i].k
		);
		f64 const measured = (f64)layeredOut[i*3+0];
		CAPTURE(i);
		CHECK(std::isfinite(measured));
		f64 const d = std::fabs(measured - expected);
		worst = std::max(worst, d);
		CHECK(d < 0.06);
	}
	MESSAGE("fuzz furnace layered vs closed-form worst |diff|: ", worst);

	ponder::zeltner_tables_destroy(tables);
}

TEST_CASE("fuzz eval furnace: heatmap (incidence x roughness, isolated fuzz)") {
	auto tables = ponder::zeltner_tables_create();

	constexpr u32 kSize = 48u;
	std::vector<f32> furnaceFlat, lookupFlat;
	std::vector<f32> cts(kSize), roughs(kSize);
	for (u32 y = 0; y < kSize; ++y) {
		roughs[y] = 0.05f + 0.95f * (f32)y / (f32)(kSize - 1);
	}
	for (u32 x = 0; x < kSize; ++x) {
		cts[x] = 0.02f + 0.98f * (f32)x / (f32)(kSize - 1);
	}
	for (u32 y = 0; y < kSize; ++y) {
		for (u32 x = 0; x < kSize; ++x) {
			f32 const ct = cts[x];
			f32 const st = std::sqrt(std::fmax(0.0f, 1.0f - ct * ct));
			push_furnace_sample(
				furnaceFlat, { st, 0.0f, ct },
				1.0f, { 1.0f, 1.0f, 1.0f }, roughs[y], { 0.0f, 0.0f, 0.0f }
			);
			lookupFlat.push_back(ct);
			lookupFlat.push_back(roughs[y]);
		}
	}
	auto const furnaceOut = run_furnace(
		tables.zeltnerLtcParamHandle, furnaceFlat, 96u, 192u
	);
	auto const lookupOut = run_lookup(tables.zeltnerLtcParamHandle, lookupFlat);

	std::vector<f32> img(kSize * kSize);
	f64 worst = 0.0;
	u32 nanCount = 0u;
	for (u32 i = 0; i < kSize * kSize; ++i) {
		f32 const measured = furnaceOut[i*3+0];
		if (!std::isfinite(measured)) { img[i] = std::nanf(""); nanCount++; continue; }
		img[i] = measured;
		f64 const d = std::fabs((f64)measured - (f64)lookupOut[i*3+2]);
		worst = std::max(worst, d);
	}
	bool const ok = test::write_heatmap_png(
		img, img, img, kSize, kSize,
		FUZZ_OUTPUT_DIR "fuzz_furnace_heatmap.png", &nanCount
	);
	CHECK(ok);
	CAPTURE(worst);
	CHECK(nanCount == 0u);

	ponder::zeltner_tables_destroy(tables);
}

} // TEST_SUITE("[headless]")
