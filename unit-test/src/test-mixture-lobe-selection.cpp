#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <ponder/zeltner-tables.hpp>
#include "util.hpp"

#include <cmath>
#include <vector>
#include <array>
#include <algorithm>

// ---------------------------------------------------------------------------
// seventh milestone in the openpbr port-verification series: the multi-lobe
// mixture machinery in lib/ponder/shaders/util-material-openpbr.glsl
// (openPbrLobeSelection, openPbrSampleWo, openPbrEvaluatePdfSelected) that
// combines however many of {coat, specular, diffuse, fuzz, transmission}
// have nonzero weight into one stochastic sampler and one pdf. every prior
// milestone tested its lobe in isolation (every other weight pinned to
// zero); this file is the first to exercise more than one lobe live at
// once. per-lobe pdf/sample correctness is out of scope here (already
// verified in the dielectric-specular, glossy-diffuse, metallic, fuzz and
// transmission milestones) -- this suite is purely about the combination:
// do the sampler's draw fractions and the pdf's weighting fractions agree,
// does the below-horizon multi-term leak branch stay finite and correctly
// weighted, and are zero-probability lobes actually excluded rather than
// relying on probability*pdf to cancel a NaN.
//
// crosscheck note: there is no materialx mixture-bsdf equivalent for
// openpbr's lobe-selection weights (materialx layers bsdfs differently), so
// the "materialx crosscheck" requirement here is satisfied by
// mixture_selection_crosscheck.comp -- an independent re-derivation of the
// *combination* arithmetic (budget split / max / sum) against the same
// already-individually-verified per-lobe weight primitives
// (utilMicrofacetDielectricAlbedo, utilZeltnerFuzzLookup). see that file's
// header comment for exactly what is and isn't independent about it.
// ---------------------------------------------------------------------------

namespace {

// diagnostic experiment: every test case in this file previously called
// ponder::zeltner_tables_create()/destroy() individually (one image +
// sampler + bindless descriptor handle allocated and freed per
// TEST_CASE, matching test-fuzz-eval.cpp's established convention). two
// of the chi-square checks below (diffuse+specular, kitchen sink) failed
// non-deterministically depending on which *other* test cases were also
// selected to run -- reproducible with a trivial fill_u32.comp-based
// harness stress test that never touches the zeltner texture at all, but
// not with a version that never creates/destroys an image+sampler. this
// shares a single long-lived handle across the whole file instead, to
// test whether the repeated create/destroy cycle (and its bindless-slot
// churn) is the actual trigger. never destroyed -- acceptable leak for a
// test-binary-lifetime diagnostic; not a fix, just a probe.
u32 shared_zeltner_handle() {
	static ponder::ZeltnerTables const tables = ponder::zeltner_tables_create();
	return tables.zeltnerLtcParamHandle;
}

// must match OPENPBR_LOBE_* in util-material-openpbr.glsl
constexpr u32 kLobeCoat = 0u;
constexpr u32 kLobeSpecular = 1u;
constexpr u32 kLobeDiffuse = 2u;
constexpr u32 kLobeFuzz = 3u;
constexpr u32 kLobeTransmission = 4u;
constexpr u32 kLobeCount = 5u;

// must match util-sampling-bins.glsl
constexpr u32 kThetaBins = 40u;
constexpr u32 kPhiBins = 80u;
constexpr u32 kBinCount = kThetaBins * kPhiBins;

// the material fields the mixture chain actually reads (coat/specular/base/
// transmission/fuzz weights and shapes); everything else in OpenPbrMaterial
// (base color, subsurface, thin film, emission, geometry textures) is
// unused by openPbrLobeSelection/openPbrSampleWo/openPbrEvaluatePdfSelected
// and left out. not a push-constant struct itself -- each shader runner
// below copies these fields into its own push layout.
struct MixtureConfig {
	f32 coatWeight = 0.0f;
	f32 coatRoughness = 0.3f;
	f32 coatRoughnessAnisotropy = 0.0f;
	f32 coatIor = 1.6f;
	f32 specularWeight = 1.0f;
	f32 specularIor = 1.5f;
	f32 specularRoughness = 0.3f;
	f32 specularRoughnessAnisotropy = 0.0f;
	f32 specularRoughnessAnisotropyRotation = 0.0f;
	f32 baseMetalness = 0.0f;
	f32 transmissionWeight = 0.0f;
	f32 geometryThinWalled = 0.0f;
	f32 fuzzWeight = 0.0f;
	f32 fuzzRoughness = 0.3f;
};

// ---------------------------------------------------------------------------
// diagnostic: poison every DeviceOnly f32 output buffer with an
// impossible-to-legitimately-produce sentinel before dispatch (every
// value these shaders write is a pdf/probability, always >= 0). a
// surviving sentinel after readback means either an unwritten slot (a
// dispatch-sizing/indexing bug in the shader) or, if the leftover value
// isn't the exact sentinel but also isn't a plausible pdf, genuine
// cross-buffer bleed-through -- see the mixture-lobe-selection chi-square
// investigation this is chasing.
// ---------------------------------------------------------------------------

constexpr f32 kPoisonSentinel = -123456.0f;

void poison_buffer_f32(vkof::Buffer const buf, u32 const count) {
	std::vector<f32> const poison(count, kPoisonSentinel);
	vkof::buffer_upload({
		.buffer = buf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(poison.data()),
			count * sizeof(f32)
		),
	});
}

// counts values still at exactly kPoisonSentinel (unwritten slots),
// separately, values that are negative but not the sentinel (a write
// happened, but to something that can never be a legitimate pdf/
// probability -- points at corrupted rather than merely-missing data), and
// separately again, NaN (a write happened to something that isn't even a
// number). NaN is its own bucket rather than folding into otherNegative
// because every comparison against NaN is false: `v < 0.0f` silently skips
// it, and chi_square_where's `expected[i] >= 5.0f` cell-inclusion test does
// the same, so a NaN expected value doesn't fail the cell -- it just drops
// out of the statistic entirely as if that bin was never sampled. that
// masked exactly this class of bug once already (the mixture-lobe-selection
// chi-square regression this scan is chasing): a genuinely broken bin
// silently passed by not being checked at all, rather than by actually
// matching
struct PoisonScan { u32 stillSentinel; u32 otherNegative; u32 nanCount; };

PoisonScan scan_for_poison(std::vector<f32> const & data) {
	PoisonScan scan { 0u, 0u, 0u };
	for (f32 const v : data) {
		if (std::isnan(v)) {
			scan.nanCount++;
		} else if (v == kPoisonSentinel) {
			scan.stillSentinel++;
		} else if (v < 0.0f) {
			scan.otherNegative++;
		}
	}
	return scan;
}

// ---------------------------------------------------------------------------
// mixture_selection_crosscheck.comp runner (task: selection-weight
// crosscheck)
// ---------------------------------------------------------------------------

struct SelectionCrosscheckPush {
	u64 inVa;
	u64 mineOutVa;
	u64 refOutVa;
	u32 zeltnerLtcParamHandle;
	u32 count;
};
static_assert(sizeof(SelectionCrosscheckPush) == 32);

struct SelResult {
	f32 coat, specular, diffuse, fuzz, transmission, total;
};

struct SelectionCrosscheckResult {
	std::vector<SelResult> mine;
	std::vector<SelResult> ref;
};

SelectionCrosscheckResult run_selection_crosscheck(
	u32 const zeltnerLtcParamHandle,
	std::vector<f32> const & configsFlat
) {
	u32 const count = (u32)configsFlat.size() / 11u;
	REQUIRE(configsFlat.size() == (size_t)count * 11u);
	static constexpr u32 kLocalSize = 64;

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "mixture_selection_crosscheck.comp",
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
	auto mineBuf = vkof::buffer_create({
		.byteCount = (u64)count * 6u * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	auto refBuf = vkof::buffer_create({
		.byteCount = (u64)count * 6u * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	poison_buffer_f32(mineBuf, count * 6u);
	poison_buffer_f32(refBuf, count * 6u);

	SelectionCrosscheckPush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.mineOutVa = vkof::buffer_virtual_address(mineBuf),
		.refOutVa = vkof::buffer_virtual_address(refBuf),
		.zeltnerLtcParamHandle = zeltnerLtcParamHandle,
		.count = count,
	};
	test::dispatch(pl, push, (count + kLocalSize - 1) / kLocalSize);

	auto mineFlat = test::readback<f32>(mineBuf, 0, count * 6u);
	auto refFlat = test::readback<f32>(refBuf, 0, count * 6u);
	auto const minePoison = scan_for_poison(mineFlat);
	auto const refPoison = scan_for_poison(refFlat);
	CAPTURE(minePoison.stillSentinel);
	CAPTURE(minePoison.otherNegative);
	CAPTURE(minePoison.nanCount);
	CAPTURE(refPoison.stillSentinel);
	CAPTURE(refPoison.otherNegative);
	CAPTURE(refPoison.nanCount);
	CHECK(minePoison.stillSentinel == 0u);
	CHECK(minePoison.otherNegative == 0u);
	CHECK(minePoison.nanCount == 0u);
	CHECK(refPoison.stillSentinel == 0u);
	CHECK(refPoison.otherNegative == 0u);
	CHECK(refPoison.nanCount == 0u);
	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(mineBuf);
	vkof::buffer_destroy(refBuf);
	vkof::pipeline_destroy(pl);

	SelectionCrosscheckResult out;
	out.mine.resize(count);
	out.ref.resize(count);
	for (u32 i = 0; i < count; ++i) {
		out.mine[i] = {
			mineFlat[i*6+0], mineFlat[i*6+1], mineFlat[i*6+2],
			mineFlat[i*6+3], mineFlat[i*6+4], mineFlat[i*6+5],
		};
		out.ref[i] = {
			refFlat[i*6+0], refFlat[i*6+1], refFlat[i*6+2],
			refFlat[i*6+3], refFlat[i*6+4], refFlat[i*6+5],
		};
	}
	return out;
}

void push_selection_config(
	std::vector<f32> & flat,
	f32 const mu, MixtureConfig const & c
) {
	flat.push_back(mu);
	flat.push_back(c.coatWeight);
	flat.push_back(c.coatRoughness);
	flat.push_back(c.coatIor);
	flat.push_back(c.specularWeight);
	flat.push_back(c.specularIor);
	flat.push_back(c.specularRoughness);
	flat.push_back(c.baseMetalness);
	flat.push_back(c.transmissionWeight);
	flat.push_back(c.fuzzWeight);
	flat.push_back(c.fuzzRoughness);
}

// ---------------------------------------------------------------------------
// mixture_histogram.comp / mixture_expected.comp runners (chi-square)
// ---------------------------------------------------------------------------

struct MixtureHistogramPush {
	u64 histogramVa;
	u64 pickCountVa;
	u32 zeltnerLtcParamHandle;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 wi;
	f32 coatWeight;
	f32 coatRoughness;
	f32 coatRoughnessAnisotropy;
	f32 coatIor;
	f32 specularWeight;
	f32 specularIor;
	f32 specularRoughness;
	f32 specularRoughnessAnisotropy;
	f32 specularRoughnessAnisotropyRotation;
	f32 baseMetalness;
	f32 transmissionWeight;
	f32 geometryThinWalled;
	f32 fuzzWeight;
	f32 fuzzRoughness;
};
static_assert(sizeof(MixtureHistogramPush) == 96);

struct MixtureExpectedPush {
	u64 expectedVa;
	u32 zeltnerLtcParamHandle;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 wi;
	f32 coatWeight;
	f32 coatRoughness;
	f32 coatRoughnessAnisotropy;
	f32 coatIor;
	f32 specularWeight;
	f32 specularIor;
	f32 specularRoughness;
	f32 specularRoughnessAnisotropy;
	f32 specularRoughnessAnisotropyRotation;
	f32 baseMetalness;
	f32 transmissionWeight;
	f32 geometryThinWalled;
	f32 fuzzWeight;
	f32 fuzzRoughness;
};
static_assert(sizeof(MixtureExpectedPush) == 88);

struct HistogramResult {
	std::vector<u32> bins;
	std::array<u32, kLobeCount> pickCounts;
};

HistogramResult run_mixture_histogram(
	u32 const zeltnerLtcParamHandle,
	f32v3 const wi,
	MixtureConfig const & c,
	u32 const sampleCount,
	u32 const seedSalt
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "mixture_histogram.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	u32 const slotCount = kBinCount + 1u;
	auto histBuf = vkof::buffer_create({
		.byteCount = slotCount * sizeof(u32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	std::vector<u32> const zerosHist(slotCount, 0u);
	vkof::buffer_upload({
		.buffer = histBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(zerosHist.data()),
			slotCount * sizeof(u32)
		),
	});
	auto pickBuf = vkof::buffer_create({
		.byteCount = kLobeCount * sizeof(u32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	std::vector<u32> const zerosPick(kLobeCount, 0u);
	vkof::buffer_upload({
		.buffer = pickBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(zerosPick.data()),
			kLobeCount * sizeof(u32)
		),
	});

	MixtureHistogramPush const push {
		.histogramVa = vkof::buffer_virtual_address(histBuf),
		.pickCountVa = vkof::buffer_virtual_address(pickBuf),
		.zeltnerLtcParamHandle = zeltnerLtcParamHandle,
		.sampleCount = sampleCount,
		.seedSalt = seedSalt,
		.wi = wi,
		.coatWeight = c.coatWeight,
		.coatRoughness = c.coatRoughness,
		.coatRoughnessAnisotropy = c.coatRoughnessAnisotropy,
		.coatIor = c.coatIor,
		.specularWeight = c.specularWeight,
		.specularIor = c.specularIor,
		.specularRoughness = c.specularRoughness,
		.specularRoughnessAnisotropy = c.specularRoughnessAnisotropy,
		.specularRoughnessAnisotropyRotation = (
			c.specularRoughnessAnisotropyRotation
		),
		.baseMetalness = c.baseMetalness,
		.transmissionWeight = c.transmissionWeight,
		.geometryThinWalled = c.geometryThinWalled,
		.fuzzWeight = c.fuzzWeight,
		.fuzzRoughness = c.fuzzRoughness,
	};
	static constexpr u32 kLocalSize = 256;
	test::dispatch(pl, push, (sampleCount + kLocalSize - 1) / kLocalSize);

	HistogramResult out;
	out.bins = test::readback<u32>(histBuf, 0, slotCount);
	auto picks = test::readback<u32>(pickBuf, 0, kLobeCount);
	for (u32 i = 0; i < kLobeCount; ++i) { out.pickCounts[i] = picks[i]; }

	vkof::buffer_destroy(histBuf);
	vkof::buffer_destroy(pickBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

std::vector<f32> run_mixture_expected(
	u32 const zeltnerLtcParamHandle,
	f32v3 const wi,
	MixtureConfig const & c,
	u32 const sampleCount
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "mixture_expected.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto buf = vkof::buffer_create({
		.byteCount = kBinCount * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	poison_buffer_f32(buf, kBinCount);
	MixtureExpectedPush const push {
		.expectedVa = vkof::buffer_virtual_address(buf),
		.zeltnerLtcParamHandle = zeltnerLtcParamHandle,
		.sampleCount = sampleCount,
		.seedSalt = 0u,
		.wi = wi,
		.coatWeight = c.coatWeight,
		.coatRoughness = c.coatRoughness,
		.coatRoughnessAnisotropy = c.coatRoughnessAnisotropy,
		.coatIor = c.coatIor,
		.specularWeight = c.specularWeight,
		.specularIor = c.specularIor,
		.specularRoughness = c.specularRoughness,
		.specularRoughnessAnisotropy = c.specularRoughnessAnisotropy,
		.specularRoughnessAnisotropyRotation = (
			c.specularRoughnessAnisotropyRotation
		),
		.baseMetalness = c.baseMetalness,
		.transmissionWeight = c.transmissionWeight,
		.geometryThinWalled = c.geometryThinWalled,
		.fuzzWeight = c.fuzzWeight,
		.fuzzRoughness = c.fuzzRoughness,
	};
	static constexpr u32 kLocalSize = 256;
	test::dispatch(pl, push, (kBinCount + kLocalSize - 1) / kLocalSize);

	auto out = test::readback<f32>(buf, 0, kBinCount);
	if (vkof::probe_message_count() > 0u) {
		MESSAGE(
			"gpu probe: ", std::string(vkof::probe_message(0u)),
			" | cpu readback out[0]=", out[0]
		);
	}
	auto const poison = scan_for_poison(out);
	CAPTURE(poison.stillSentinel);
	CAPTURE(poison.otherNegative);
	CAPTURE(poison.nanCount);
	CHECK(poison.stillSentinel == 0u);
	CHECK(poison.otherNegative == 0u);
	CHECK(poison.nanCount == 0u);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

// ---------------------------------------------------------------------------
// chi-square (same pearson + wilson-hilferty convention as every prior
// milestone; see test-transmission-sampling.cpp). generalized to run over
// an arbitrary bin-index predicate so the below-horizon leak branch can be
// isolated from the full-sphere aggregate.
// ---------------------------------------------------------------------------

struct ChiSquare {
	f64 statistic;
	f64 dof;
	f64 pValue;
	u64 observedTotal;
	f64 expectedTotal;
	f64 pooledObserved;
	f64 pooledExpected;
	// a NaN expected[i] fails both `expected[i] >= 5.0f` (comparisons
	// against NaN are always false) and the pooled-bucket path (pooledE
	// would itself become NaN and, since pooled never folds back into
	// statistic, would then just be silently ignored too) -- either way a
	// NaN bin drops out of the test instead of failing it. counted
	// separately here so callers can CHECK it's zero and turn "silently not
	// checked" into an honest failure
	u32 nanExpectedCount;
};

template <typename Predicate>
ChiSquare chi_square_where(
	std::vector<u32> const & observed,
	std::vector<f32> const & expected,
	Predicate && include
) {
	f64 statistic = 0.0;
	u32 cells = 0u;
	f64 pooledO = 0.0;
	f64 pooledE = 0.0;
	u64 observedTotal = 0u;
	f64 expectedTotal = 0.0;
	u32 nanExpectedCount = 0u;
	for (u32 i = 0; i < kBinCount; ++i) {
		if (!include(i)) { continue; }
		observedTotal += observed[i];
		if (std::isnan(expected[i])) {
			nanExpectedCount++;
			continue;
		}
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
		statistic, dof, pValue, observedTotal, expectedTotal, pooledO, pooledE,
		nanExpectedCount
	};
}

ChiSquare chi_square(
	std::vector<u32> const & observed, std::vector<f32> const & expected
) {
	return chi_square_where(observed, expected, [](u32) { return true; });
}

bool bin_is_below_horizon(u32 const bin) {
	// bsdfVerifyDirectionToBin: thetaBin = bin / kPhiBins, theta=pi/2 is
	// exactly thetaBin==20 (bin count 40 over [0,pi]); bin 20 straddles the
	// horizon itself (theta in [pi/2, pi/2+pi/40)) so it's excluded from
	// both halves rather than arbitrarily assigned to one
	u32 const thetaBin = bin / kPhiBins;
	return thetaBin > kThetaBins / 2u;
}
bool bin_is_above_horizon(u32 const bin) {
	u32 const thetaBin = bin / kPhiBins;
	return thetaBin < kThetaBins / 2u;
}

// full sampler-vs-pdf statistical check for one mixture configuration.
// reports (does not gate on) the above/below-horizon split so a below-
// horizon-only regression -- the leak branch's top suspect location per
// this milestone's brief -- doesn't hide inside a healthy aggregate
// p-value.
void check_mixture_chi_square(
	u32 const zeltnerLtcParamHandle,
	char const * const label,
	f32v3 const wi,
	MixtureConfig const & c,
	u32 const seedSalt,
	char const * const residualPngPath = nullptr
) {
	constexpr u32 kSampleCount = 1u << 20;
	auto const hist = (
		run_mixture_histogram(zeltnerLtcParamHandle, wi, c, kSampleCount, seedSalt)
	);
	auto const expected = (
		run_mixture_expected(zeltnerLtcParamHandle, wi, c, kSampleCount)
	);

	CAPTURE(label);
	CAPTURE(wi.z);

	// no sample may be non-finite or non-unit
	CHECK(hist.bins[kBinCount] == 0u);
	// every expected-side cell must itself be finite (a NaN here would
	// silently zero out of the E>=5 pearson sum instead of failing loudly)
	u32 nonFiniteExpected = 0u;
	for (f32 const e : expected) { if (!std::isfinite(e)) { nonFiniteExpected++; } }
	CHECK(nonFiniteExpected == 0u);

	// diagnostic: top-5 bins by (expected - observed), to see whether the
	// aggregate expectedTotal excess concentrates in a handful of outlier
	// bins (points at a specific corrupted write) or is spread smoothly
	// (points at a genuine, broad quadrature/formula effect)
	{
		std::vector<u32> order(kBinCount);
		for (u32 i = 0; i < kBinCount; ++i) { order[i] = i; }
		std::sort(order.begin(), order.end(), [&](u32 const a, u32 const b) {
			return (
				(expected[a] - (f32)hist.bins[a])
				> (expected[b] - (f32)hist.bins[b])
			);
		});
		for (u32 rank = 0; rank < 5; ++rank) {
			u32 const bin = order[rank];
			u32 const thetaBin = bin / kPhiBins;
			u32 const phiBin = bin % kPhiBins;
			MESSAGE(
				label, ": outlier #", rank, " bin=", bin,
				" thetaBin=", thetaBin, " phiBin=", phiBin,
				" expected=", expected[bin], " observed=", hist.bins[bin]
			);
		}
	}

	auto const cs = chi_square(hist.bins, expected);
	CAPTURE(cs.statistic);
	CAPTURE(cs.dof);
	CAPTURE(cs.pooledObserved);
	CAPTURE(cs.pooledExpected);
	CAPTURE(cs.nanExpectedCount);
	CHECK(cs.nanExpectedCount == 0u);
	CHECK(cs.observedTotal == kSampleCount);
	CHECK(
		cs.expectedTotal == doctest::Approx((f64)kSampleCount).epsilon(0.02)
	);
	CHECK(cs.pValue > 1e-4);
	// bounded vndf support-cut curves and the transmission refraction cone
	// both cut through bin interiors on this grid (see every prior
	// milestone's sweep tests); a mixture of several such lobes compounds
	// the flicker sources, so the pooled allowance is generous
	CHECK(cs.pooledObserved <= cs.pooledExpected + 400.0);

	// below-horizon-only split: isolates openPbrEvaluatePdfSelected's
	// dotNorWo<=0 leak branch (up to three guarded terms summed) from the
	// (already well-covered by prior milestones) above-horizon aggregate
	auto const csBelow = chi_square_where(hist.bins, expected, bin_is_below_horizon);
	auto const csAbove = chi_square_where(hist.bins, expected, bin_is_above_horizon);
	CAPTURE(csBelow.statistic);
	CAPTURE(csBelow.dof);
	CAPTURE(csBelow.pooledObserved);
	CAPTURE(csBelow.pooledExpected);
	CAPTURE(csBelow.nanExpectedCount);
	CAPTURE(csAbove.nanExpectedCount);
	MESSAGE(
		label, ": below-horizon p=", csBelow.pValue,
		" (n=", csBelow.observedTotal, "), above-horizon p=", csAbove.pValue,
		" (n=", csAbove.observedTotal, ")"
	);
	CHECK(csBelow.nanExpectedCount == 0u);
	CHECK(csAbove.nanExpectedCount == 0u);
	if (csBelow.observedTotal > 0u) {
		CHECK(csBelow.pValue > 1e-4);
		CHECK(csBelow.pooledObserved <= csBelow.pooledExpected + 200.0);
	}

	// mixture invariant: the empirical fraction of draws that picked each
	// lobe must match that lobe's sel.probabilityX / probabilityTotal --
	// sampler and pdf are required to use the identical fractions or they
	// silently disagree. sel itself comes from
	// mixture_selection_crosscheck.comp's "mine" output (the real
	// openPbrLobeSelection), not re-derived here.
	{
		std::vector<f32> selFlat;
		push_selection_config(selFlat, wi.z, c);
		auto const sel = (
			run_selection_crosscheck(zeltnerLtcParamHandle, selFlat).mine[0]
		);
		f32 const total = sel.total;
		CAPTURE(total);
		if (total > 1e-6f) {
			std::array<f32, kLobeCount> const expectedFraction {
				sel.coat / total, sel.specular / total, sel.diffuse / total,
				sel.fuzz / total, sel.transmission / total,
			};
			char const * const lobeNames[kLobeCount] = {
				"coat", "specular", "diffuse", "fuzz", "transmission"
			};
			f64 pickChiSq = 0.0;
			for (u32 lobe = 0; lobe < kLobeCount; ++lobe) {
				f64 const e = (f64)expectedFraction[lobe] * (f64)kSampleCount;
				f64 const o = (f64)hist.pickCounts[lobe];
				CAPTURE(lobeNames[lobe]);
				CAPTURE(e);
				CAPTURE(o);
				// a lobe with genuinely zero selection probability must
				// never be picked, exactly -- not a statistical bound
				if (e < 1e-6) {
					CHECK(o == 0.0);
				} else if (e >= 5.0) {
					pickChiSq += (o - e) * (o - e) / e;
				}
			}
			// 5 categories, 4 degrees of freedom; p < 1e-4 threshold at
			// dof=4 is statistic > ~23.5, i.e. this only trips on a real
			// fraction mismatch, not sampling noise at 1M draws
			CAPTURE(pickChiSq);
			CHECK(pickChiSq < 30.0);
		}
	}

	if (residualPngPath != nullptr) {
		std::vector<f32> r(kBinCount), g(kBinCount, 0.0f), b(kBinCount);
		for (u32 i = 0; i < kBinCount; ++i) {
			f32 const e = expected[i];
			f32 const res = (
				((f32)hist.bins[i] - e) / std::sqrt(std::fmax(e, 1.0f))
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

// ---------------------------------------------------------------------------
// mixture_edge_sweep.comp runner (zero-probability guard)
// ---------------------------------------------------------------------------

struct EdgeSweepPush {
	u64 outVa;
	u32 width;
	u32 height;
	u32 excludedLobe;
	u32 zeltnerLtcParamHandle;
	f32 phiWo;
	f32 coatWeight;
	f32 coatRoughness;
	f32 coatRoughnessAnisotropy;
	f32 coatIor;
	f32 specularWeight;
	f32 specularIor;
	f32 specularRoughness;
	f32 specularRoughnessAnisotropy;
	f32 specularRoughnessAnisotropyRotation;
	f32 baseMetalness;
	f32 transmissionWeight;
	f32 geometryThinWalled;
	f32 fuzzWeight;
	f32 fuzzRoughness;
};
// 84 logical bytes, padded to 88 for u64's 8-byte struct alignment
static_assert(sizeof(EdgeSweepPush) == 88);

// 4 floats per texel: mixturePdf, rawExcludedPdf, mixtureFinite, rawFinite
std::vector<f32> run_edge_sweep(
	u32 const zeltnerLtcParamHandle,
	u32 const excludedLobe,
	f32 const phiWo,
	MixtureConfig const & c,
	u32 const width,
	u32 const height
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "mixture_edge_sweep.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto buf = vkof::buffer_create({
		.byteCount = (u64)width * height * 4u * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	poison_buffer_f32(buf, width * height * 4u);
	EdgeSweepPush const push {
		.outVa = vkof::buffer_virtual_address(buf),
		.width = width,
		.height = height,
		.excludedLobe = excludedLobe,
		.zeltnerLtcParamHandle = zeltnerLtcParamHandle,
		.phiWo = phiWo,
		.coatWeight = c.coatWeight,
		.coatRoughness = c.coatRoughness,
		.coatRoughnessAnisotropy = c.coatRoughnessAnisotropy,
		.coatIor = c.coatIor,
		.specularWeight = c.specularWeight,
		.specularIor = c.specularIor,
		.specularRoughness = c.specularRoughness,
		.specularRoughnessAnisotropy = c.specularRoughnessAnisotropy,
		.specularRoughnessAnisotropyRotation = (
			c.specularRoughnessAnisotropyRotation
		),
		.baseMetalness = c.baseMetalness,
		.transmissionWeight = c.transmissionWeight,
		.geometryThinWalled = c.geometryThinWalled,
		.fuzzWeight = c.fuzzWeight,
		.fuzzRoughness = c.fuzzRoughness,
	};
	test::dispatch(pl, push, (width + 7u) / 8u, (height + 7u) / 8u);

	auto out = test::readback<f32>(buf, 0, (u64)width * height * 4u);
	auto const poison = scan_for_poison(out);
	CAPTURE(poison.stillSentinel);
	CAPTURE(poison.otherNegative);
	CAPTURE(poison.nanCount);
	CHECK(poison.stillSentinel == 0u);
	CHECK(poison.otherNegative == 0u);
	CHECK(poison.nanCount == 0u);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

} // namespace

TEST_SUITE("[headless]") {

// -----------------------------------------------------------------------------
// task: selection-weight combination crosscheck (no materialx mixture-bsdf
// equivalent exists; see file header and mixture_selection_crosscheck.comp)
// -----------------------------------------------------------------------------

TEST_CASE("mixture: lobe selection weights match independently re-derived combination") {

	std::vector<f32> flat;
	std::vector<f32> mus = { 0.02f, 0.1f, 0.3f, 0.6f, 0.9f, 1.0f };
	std::vector<MixtureConfig> configs;
	// sweep coat/specular/metal/transmission/fuzz combinations, including
	// all-off and all-on extremes
	for (f32 const coatW : { 0.0f, 0.5f, 1.0f }) {
		for (f32 const metal : { 0.0f, 0.5f, 1.0f }) {
			for (f32 const transW : { 0.0f, 0.5f, 1.0f }) {
				for (f32 const fuzzW : { 0.0f, 0.5f, 1.0f }) {
					MixtureConfig c;
					c.coatWeight = coatW;
					c.coatRoughness = 0.25f;
					c.baseMetalness = metal;
					c.transmissionWeight = transW;
					c.fuzzWeight = fuzzW;
					c.fuzzRoughness = 0.4f;
					c.specularRoughness = 0.2f;
					configs.push_back(c);
				}
			}
		}
	}
	for (f32 const mu : mus) {
		for (auto const & c : configs) { push_selection_config(flat, mu, c); }
	}

	auto const result = (
		run_selection_crosscheck(shared_zeltner_handle(), flat)
	);
	REQUIRE(result.mine.size() == result.ref.size());
	for (size_t i = 0; i < result.mine.size(); ++i) {
		CAPTURE(i);
		auto const & m = result.mine[i];
		auto const & r = result.ref[i];
		CHECK(m.coat == doctest::Approx(r.coat).epsilon(1e-4));
		CHECK(m.specular == doctest::Approx(r.specular).epsilon(1e-4));
		CHECK(m.diffuse == doctest::Approx(r.diffuse).epsilon(1e-4));
		CHECK(m.fuzz == doctest::Approx(r.fuzz).epsilon(1e-4));
		CHECK(m.transmission == doctest::Approx(r.transmission).epsilon(1e-4));
		CHECK(m.total == doctest::Approx(r.total).epsilon(1e-4));
		// sanity: total is exactly the sum of the five, both sides
		CHECK(
			m.total
			== doctest::Approx(m.coat + m.specular + m.diffuse + m.fuzz + m.transmission)
				.epsilon(1e-5)
		);
	}

}

// -----------------------------------------------------------------------------
// task: pairwise mixture chi-square (narrows a failure down to which pair,
// before the full five-lobe kitchen sink)
// -----------------------------------------------------------------------------

TEST_CASE("mixture: sampler vs pdf chi-square, diffuse+specular") {
	MixtureConfig c;
	c.specularWeight = 1.0f;
	c.specularRoughness = 0.3f;
	// diffuse gets whatever's left of the base budget; metal=0, trans=0
	for (f32 const ct : { 0.9f, 0.5f, 0.15f }) {
		std::string const png = (
			std::string(MIXTURE_OUTPUT_DIR) + "mixture_diffuse_specular_chi2_ct"
			+ std::to_string((int)(ct * 100)) + ".png"
		);
		check_mixture_chi_square(
			shared_zeltner_handle(), "diffuse+specular",
			{ std::sqrt(1.0f - ct * ct), 0.0f, ct }, c, 1u, png.c_str()
		);
	}
}

TEST_CASE("DEBUG: single point breakdown, diffuse+specular near mirror") {
	struct DebugPush {
		u64 outVa;
		u32 zeltnerLtcParamHandle;
		f32v3 wi;
		f32v3 wo;
		f32 coatWeight;
		f32 coatRoughness;
		f32 coatIor;
		f32 specularWeight;
		f32 specularIor;
		f32 specularRoughness;
		f32 baseMetalness;
		f32 transmissionWeight;
		f32 fuzzWeight;
		f32 fuzzRoughness;
	};
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "mixture_debug_point.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);
	auto buf = vkof::buffer_create({
		.byteCount = 10 * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	f32 const ct = 0.9f;
	f32v3 const wi = { std::sqrt(1.0f - ct * ct), 0.0f, ct };
	// near the specular mirror direction of wi
	f32v3 const wo = { -wi.x, -wi.y, wi.z };
	DebugPush const push {
		.outVa = vkof::buffer_virtual_address(buf),
		.zeltnerLtcParamHandle = shared_zeltner_handle(),
		.wi = wi,
		.wo = wo,
		.coatWeight = 0.0f,
		.coatRoughness = 0.3f,
		.coatIor = 1.6f,
		.specularWeight = 1.0f,
		.specularIor = 1.5f,
		.specularRoughness = 0.3f,
		.baseMetalness = 0.0f,
		.transmissionWeight = 0.0f,
		.fuzzWeight = 0.0f,
		.fuzzRoughness = 0.3f,
	};
	test::dispatch(pl, push, 1u);
	auto out = test::readback<f32>(buf, 0, 10);
	MESSAGE(
		"probCoat=", out[0], " probSpec=", out[1], " probDiff=", out[2],
		" probFuzz=", out[3], " probTrans=", out[4], " total=", out[5]
	);
	MESSAGE(
		"specPdf=", out[6], " diffPdf=", out[7], " mixturePdf(lib)=", out[8],
		" mixturePdf(handRecombined)=", out[9]
	);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
}

TEST_CASE("DEBUG: sub-grid scan, transmission+specular outlier bin") {
	struct DebugPush {
		u64 outVa;
		u32 zeltnerLtcParamHandle;
		f32v3 wi;
		f32 theta0;
		f32 theta1;
		f32 phi0;
		f32 phi1;
		f32 coatWeight;
		f32 coatRoughness;
		f32 coatIor;
		f32 specularWeight;
		f32 specularIor;
		f32 specularRoughness;
		f32 baseMetalness;
		f32 transmissionWeight;
		f32 fuzzWeight;
		f32 fuzzRoughness;
	};
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "mixture_debug_point.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);
	auto buf = vkof::buffer_create({
		.byteCount = 10 * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	f32 const ct = 0.15f;
	f32v3 const wi = { std::sqrt(1.0f - ct * ct), 0.0f, ct };
	// bin=1360, thetaBin=17, phiBin=0 -- the worst outlier from the
	// transmission+specular chi-square failure at ct=0.15 (kThetaBins=40,
	// kPhiBins=80)
	constexpr f32 kPi = 3.14159265f;
	DebugPush const push {
		.outVa = vkof::buffer_virtual_address(buf),
		.zeltnerLtcParamHandle = shared_zeltner_handle(),
		.wi = wi,
		.theta0 = kPi * 17.0f / 40.0f,
		.theta1 = kPi * 18.0f / 40.0f,
		.phi0 = 2.0f * kPi * 0.0f / 80.0f,
		.phi1 = 2.0f * kPi * 1.0f / 80.0f,
		.coatWeight = 0.0f,
		.coatRoughness = 0.3f,
		.coatIor = 1.6f,
		.specularWeight = 1.0f,
		.specularIor = 1.5f,
		.specularRoughness = 0.2f,
		.baseMetalness = 0.0f,
		.transmissionWeight = 1.0f,
		.fuzzWeight = 0.0f,
		.fuzzRoughness = 0.3f,
	};
	test::dispatch(pl, push, 1u);
	auto out = test::readback<f32>(buf, 0, 10);
	MESSAGE(
		"maxMixturePdf=", out[0], " at theta=", out[1], " phi=", out[2]
	);
	MESSAGE(
		"at max: transmissionTirPdf=", out[3], " specPdf=", out[4],
		" dotWiH=", out[5], " len(wi+wo)=", out[6]
	);
	MESSAGE(
		"probSpec=", out[7], " probTrans=", out[8], " total=", out[9]
	);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
}

TEST_CASE("DEBUG: specular alone via baseMetalness=1 isolation") {
	MixtureConfig c;
	c.specularWeight = 1.0f;
	c.specularRoughness = 0.3f;
	c.baseMetalness = 1.0f;
	f32 const ct = 0.15f;
	check_mixture_chi_square(
		shared_zeltner_handle(), "specular-alone-debug",
		{ std::sqrt(1.0f - ct * ct), 0.0f, ct }, c, 1u
	);
}

// does specPdf's own bounded-vndf formula, entirely alone (no
// transmission at all), already fail a chi-square at the grazing-
// retroreflection configuration the debug sub-grid scan found (wi and wo
// both near-horizon, mirrored across the normal)? same roughness/ior as
// the failing transmission+specular pairwise test, same three wi angles.
// if this fails too, the bounded technique itself has a gap there; if it
// passes, the earlier failure is specific to summing two lobes that both
// carry mass into the same region (or a quadrature-resolution artifact).
TEST_CASE("mixture: specular alone, same params as failing transmission+specular") {
	MixtureConfig c;
	c.specularWeight = 1.0f;
	c.specularRoughness = 0.2f;
	c.specularIor = 1.5f;
	c.baseMetalness = 1.0f;
	for (f32 const ct : { 0.9f, 0.5f, 0.15f }) {
		check_mixture_chi_square(
			shared_zeltner_handle(), "specular-alone-matched-params",
			{ std::sqrt(1.0f - ct * ct), 0.0f, ct }, c, 13u
		);
	}
}

// isolates the kitchen-sink chi-square outlier pattern: expected mass
// concentrated 2-2.7x too high at phiBin=0/79 (the phi=0/2pi wraparound
// seam, which sits exactly in wi's azimuthal plane), only observed in
// configs with nonzero specular anisotropy. this pins the anisotropic
// specular lobe alone (baseMetalness=1 zeros probabilityDiffuse, no
// coat/fuzz/transmission) at the same roughness/anisotropy/rotation as
// the kitchen-sink config, run as its own dedicated test case -- if the
// seam blowup reproduces here, it's inherent to the anisotropic lobe
// itself, independent of the mixture combination or of which other test
// cases happen to run alongside it.
TEST_CASE("mixture: anisotropic specular alone, phi=0 seam isolation") {
	MixtureConfig c;
	c.specularWeight = 1.0f;
	c.specularRoughness = 0.25f;
	c.specularRoughnessAnisotropy = 0.6f;
	c.specularRoughnessAnisotropyRotation = 0.7f;
	c.specularIor = 1.5f;
	c.baseMetalness = 1.0f;
	for (f32 const ct : { 0.9f, 0.5f, 0.15f }) {
		check_mixture_chi_square(
			shared_zeltner_handle(), "aniso-specular-alone",
			{ std::sqrt(1.0f - ct * ct), 0.0f, ct }, c, 6u
		);
	}
}

// bisection: pair the same anisotropic specular lobe from the isolation
// test above with exactly one of kitchen sink's other four lobes at a
// time, to find which pairing (if any short of the full five-lobe mix)
// reproduces the phi=0 seam blowup.

TEST_CASE("mixture: aniso specular seam bisection, +diffuse") {
	MixtureConfig c;
	c.specularWeight = 1.0f;
	c.specularRoughness = 0.25f;
	c.specularRoughnessAnisotropy = 0.6f;
	c.specularRoughnessAnisotropyRotation = 0.7f;
	c.specularIor = 1.5f;
	for (f32 const ct : { 0.9f, 0.5f, 0.15f }) {
		check_mixture_chi_square(
			shared_zeltner_handle(), "aniso-specular+diffuse",
			{ std::sqrt(1.0f - ct * ct), 0.0f, ct }, c, 7u
		);
	}
}

TEST_CASE("mixture: aniso specular seam bisection, +coat") {
	MixtureConfig c;
	c.coatWeight = 0.6f;
	c.coatRoughness = 0.2f;
	c.coatIor = 1.6f;
	c.specularWeight = 1.0f;
	c.specularRoughness = 0.25f;
	c.specularRoughnessAnisotropy = 0.6f;
	c.specularRoughnessAnisotropyRotation = 0.7f;
	c.specularIor = 1.5f;
	c.baseMetalness = 1.0f;
	for (f32 const ct : { 0.9f, 0.5f, 0.15f }) {
		check_mixture_chi_square(
			shared_zeltner_handle(), "aniso-specular+coat",
			{ std::sqrt(1.0f - ct * ct), 0.0f, ct }, c, 8u
		);
	}
}

TEST_CASE("mixture: aniso specular seam bisection, +transmission") {
	MixtureConfig c;
	c.specularWeight = 1.0f;
	c.specularRoughness = 0.25f;
	c.specularRoughnessAnisotropy = 0.6f;
	c.specularRoughnessAnisotropyRotation = 0.7f;
	c.specularIor = 1.5f;
	c.transmissionWeight = 1.0f;
	for (f32 const ct : { 0.9f, 0.5f, 0.15f }) {
		check_mixture_chi_square(
			shared_zeltner_handle(), "aniso-specular+transmission",
			{ std::sqrt(1.0f - ct * ct), 0.0f, ct }, c, 9u
		);
	}
}

// same +transmission pairing, rotation zeroed -- isolates whether the
// anisotropy rotation specifically is needed, or plain (unrotated)
// anisotropy already triggers it paired with transmission
TEST_CASE("mixture: aniso specular seam bisection, +transmission no rotation") {
	MixtureConfig c;
	c.specularWeight = 1.0f;
	c.specularRoughness = 0.25f;
	c.specularRoughnessAnisotropy = 0.6f;
	c.specularRoughnessAnisotropyRotation = 0.0f;
	c.specularIor = 1.5f;
	c.transmissionWeight = 1.0f;
	for (f32 const ct : { 0.9f, 0.5f, 0.15f }) {
		check_mixture_chi_square(
			shared_zeltner_handle(), "aniso-specular+transmission-norot",
			{ std::sqrt(1.0f - ct * ct), 0.0f, ct }, c, 11u
		);
	}
}

// same pairing, both anisotropy and rotation zeroed (pure isotropic
// specular + transmission at the same roughness/ior) -- clean control:
// this should match the shape of the original transmission+specular
// pairwise test, which already passed, so this should too
TEST_CASE("mixture: aniso specular seam bisection, +transmission isotropic control") {
	MixtureConfig c;
	c.specularWeight = 1.0f;
	c.specularRoughness = 0.25f;
	c.specularRoughnessAnisotropy = 0.0f;
	c.specularRoughnessAnisotropyRotation = 0.0f;
	c.specularIor = 1.5f;
	c.transmissionWeight = 1.0f;
	for (f32 const ct : { 0.9f, 0.5f, 0.15f }) {
		check_mixture_chi_square(
			shared_zeltner_handle(), "aniso-specular+transmission-isocontrol",
			{ std::sqrt(1.0f - ct * ct), 0.0f, ct }, c, 12u
		);
	}
}

TEST_CASE("mixture: aniso specular seam bisection, +fuzz") {
	MixtureConfig c;
	c.specularWeight = 1.0f;
	c.specularRoughness = 0.25f;
	c.specularRoughnessAnisotropy = 0.6f;
	c.specularRoughnessAnisotropyRotation = 0.7f;
	c.specularIor = 1.5f;
	c.baseMetalness = 1.0f;
	c.fuzzWeight = 0.4f;
	c.fuzzRoughness = 0.35f;
	for (f32 const ct : { 0.9f, 0.5f, 0.15f }) {
		check_mixture_chi_square(
			shared_zeltner_handle(), "aniso-specular+fuzz",
			{ std::sqrt(1.0f - ct * ct), 0.0f, ct }, c, 10u
		);
	}
}

TEST_CASE("mixture: sampler vs pdf chi-square, coat+specular") {
	MixtureConfig c;
	c.coatWeight = 1.0f;
	c.coatRoughness = 0.15f;
	c.coatIor = 1.6f;
	c.specularWeight = 1.0f;
	c.specularRoughness = 0.25f;
	c.baseMetalness = 1.0f;
	for (f32 const ct : { 0.9f, 0.5f, 0.15f }) {
		std::string const png = (
			std::string(MIXTURE_OUTPUT_DIR) + "mixture_coat_specular_chi2_ct"
			+ std::to_string((int)(ct * 100)) + ".png"
		);
		check_mixture_chi_square(
			shared_zeltner_handle(), "coat+specular",
			{ std::sqrt(1.0f - ct * ct), 0.0f, ct }, c, 2u, png.c_str()
		);
	}
}

TEST_CASE("mixture: sampler vs pdf chi-square, fuzz+diffuse") {
	MixtureConfig c;
	c.fuzzWeight = 1.0f;
	c.fuzzRoughness = 0.4f;
	c.specularWeight = 0.0f;
	for (f32 const ct : { 0.9f, 0.5f, 0.15f }) {
		std::string const png = (
			std::string(MIXTURE_OUTPUT_DIR) + "mixture_fuzz_diffuse_chi2_ct"
			+ std::to_string((int)(ct * 100)) + ".png"
		);
		check_mixture_chi_square(
			shared_zeltner_handle(), "fuzz+diffuse",
			{ std::sqrt(1.0f - ct * ct), 0.0f, ct }, c, 3u, png.c_str()
		);
	}
}

TEST_CASE("mixture: sampler vs pdf chi-square, transmission+specular") {
	MixtureConfig c;
	c.specularWeight = 1.0f;
	c.specularRoughness = 0.2f;
	c.specularIor = 1.5f;
	c.transmissionWeight = 1.0f;
	for (f32 const ct : { 0.9f, 0.5f, 0.15f }) {
		std::string const png = (
			std::string(MIXTURE_OUTPUT_DIR) + "mixture_transmission_specular_chi2_ct"
			+ std::to_string((int)(ct * 100)) + ".png"
		);
		check_mixture_chi_square(
			shared_zeltner_handle(), "transmission+specular",
			{ std::sqrt(1.0f - ct * ct), 0.0f, ct }, c, 4u, png.c_str()
		);
	}
}

// -----------------------------------------------------------------------------
// task: kitchen sink (all five lobes live at once, including coat+specular+
// anisotropy coupling per this milestone's brief)
// -----------------------------------------------------------------------------

TEST_CASE("mixture: sampler vs pdf chi-square, all five lobes (kitchen sink)") {
	MixtureConfig c;
	c.coatWeight = 0.6f;
	c.coatRoughness = 0.2f;
	c.coatIor = 1.6f;
	c.specularWeight = 1.0f;
	c.specularRoughness = 0.25f;
	c.specularRoughnessAnisotropy = 0.6f;
	c.specularRoughnessAnisotropyRotation = 0.7f;
	c.specularIor = 1.5f;
	c.baseMetalness = 0.0f;
	c.transmissionWeight = 0.3f;
	c.fuzzWeight = 0.4f;
	c.fuzzRoughness = 0.35f;
	for (f32 const ct : { 0.9f, 0.5f, 0.15f }) {
		std::string const png = (
			std::string(MIXTURE_OUTPUT_DIR) + "mixture_kitchen_sink_chi2_ct"
			+ std::to_string((int)(ct * 100)) + ".png"
		);
		check_mixture_chi_square(
			shared_zeltner_handle(), "kitchen-sink",
			{ std::sqrt(1.0f - ct * ct), 0.0f, ct }, c, 5u, png.c_str()
		);
	}
}

// -----------------------------------------------------------------------------
// task: zero-probability lobe exclusion. for each lobe in turn, pin its
// weight to exactly zero (with the other four live) and sweep a dense
// (grazing wi) x (full-sphere wo) grid; the mixture pdf must stay finite
// everywhere even where that lobe's own unguarded pdf primitive would not.
// -----------------------------------------------------------------------------

TEST_CASE("mixture: zero-probability lobes are excluded, not just multiplied by zero") {
	MixtureConfig c;
	c.coatWeight = 0.6f;
	c.coatRoughness = 0.15f;
	c.coatIor = 1.6f;
	c.specularWeight = 1.0f;
	c.specularRoughness = 0.2f;
	c.specularIor = 1.5f;
	c.baseMetalness = 0.0f;
	c.transmissionWeight = 0.4f;
	c.fuzzWeight = 0.5f;
	c.fuzzRoughness = 0.3f;

	struct Case { u32 lobe; char const * name; };
	std::vector<Case> const cases = {
		{ kLobeCoat, "coat" }, { kLobeSpecular, "specular" },
		{ kLobeDiffuse, "diffuse" }, { kLobeFuzz, "fuzz" },
		{ kLobeTransmission, "transmission" },
	};
	constexpr u32 kWidth = 256u;
	constexpr u32 kHeight = 256u;
	for (auto const & cs : cases) {
		for (f32 const phiWo : { 0.0f, 1.5707963f }) {
			CAPTURE(cs.name);
			CAPTURE(phiWo);
			auto const flat = (
				run_edge_sweep(
					shared_zeltner_handle(), cs.lobe, phiWo, c, kWidth, kHeight
				)
			);
			u32 nonFiniteMixture = 0u;
			u32 rawWouldHaveBeenNonFinite = 0u;
			for (u32 i = 0; i < kWidth * kHeight; ++i) {
				f32 const mixturePdf = flat[i*4+0];
				f32 const rawPdf = flat[i*4+1];
				f32 const mixtureFinite = flat[i*4+2];
				f32 const rawFinite = flat[i*4+3];
				if (mixtureFinite < 0.5f || !std::isfinite(mixturePdf)) {
					nonFiniteMixture++;
				}
				if (rawFinite < 0.5f || !std::isfinite(rawPdf)) {
					rawWouldHaveBeenNonFinite++;
				}
			}
			CAPTURE(nonFiniteMixture);
			CAPTURE(rawWouldHaveBeenNonFinite);
			// the guarded mixture pdf must never go non-finite, regardless
			// of whether the excluded lobe's raw primitive does
			CHECK(nonFiniteMixture == 0u);
			MESSAGE(
				"excluded lobe ", cs.name, " phiWo=", phiWo, ": raw primitive",
				" was non-finite at ", rawWouldHaveBeenNonFinite, " / ",
				kWidth * kHeight, " swept (wi, wo) pairs"
			);
		}
	}
}

} // TEST_SUITE("[headless]")
