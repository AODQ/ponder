#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// sampler/pdf verification for the openpbr transmission lobe: the
// production draw path (utilMicrofacetSampleGgxVndf half vector -> refract,
// with the total-internal-reflection reflect fallback) against
// openPbrTransmissionPdf / openPbrTransmissionTirReflectPdf, following the
// openpbr port workflow: chi-square sampler-vs-pdf statistics on the shared
// util-sampling-bins.glsl grid (both entering and exiting the medium),
// pdf sphere-normalization integrals with the raw-vs-restricted support
// split, single-scatter transmittance furnace checks, dense NaN sweeps,
// and heatmap visualizations. the closed-form/crosscheck side lives in
// test-transmission.cpp.
// ---------------------------------------------------------------------------

namespace {

// must match util-sampling-bins.glsl
constexpr u32 kThetaBins = 40u;
constexpr u32 kPhiBins = 80u;
constexpr u32 kBinCount = kThetaBins * kPhiBins;

// must match sampling_histogram.comp / sampling_expected.comp
constexpr u32 kModeTransmissionEnter = 4u;
constexpr u32 kModeTransmissionExit = 5u;

// layout must match the scalar push block in sampling_histogram.comp /
// sampling_expected.comp. the transmission modes repurpose alpha as
// (specularRoughness, ior)
struct SamplingStatsPush {
	u64 bufferVa;
	u32 mode;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 wi;
	f32v2 alpha;
	// (specularRoughnessAnisotropy, rotation)
	f32v2 aniso;
};
static_assert(sizeof(SamplingStatsPush) == 48);

// observed side: histogram over the sphere bins; slot kBinCount counts
// non-finite / non-unit sampled directions
std::vector<u32> run_histogram(
	u32 const mode,
	f32v3 const wi,
	f32v2 const alpha,
	f32v2 const aniso,
	u32 const sampleCount,
	u32 const seedSalt
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "sampling_histogram.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	u32 const slotCount = kBinCount + 1u;
	auto buf = vkof::buffer_create({
		.byteCount = slotCount * sizeof(u32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	std::vector<u32> const zeros(slotCount, 0u);
	vkof::buffer_upload({
		.buffer = buf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(zeros.data()),
			slotCount * sizeof(u32)
		),
	});

	SamplingStatsPush const push {
		.bufferVa = vkof::buffer_virtual_address(buf),
		.mode = mode,
		.sampleCount = sampleCount,
		.seedSalt = seedSalt,
		.wi = wi,
		.alpha = alpha,
		.aniso = aniso,
	};
	static constexpr u32 kLocalSize = 256;
	test::dispatch(pl, push, (sampleCount + kLocalSize - 1) / kLocalSize);

	auto out = test::readback<u32>(buf, 0, slotCount);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

// expected side: per-bin quadrature of the mode's pdf, scaled by
// sampleCount
std::vector<f32> run_expected(
	u32 const mode,
	f32v3 const wi,
	f32v2 const alpha,
	f32v2 const aniso,
	u32 const sampleCount
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "sampling_expected.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto buf = vkof::buffer_create({
		.byteCount = kBinCount * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	SamplingStatsPush const push {
		.bufferVa = vkof::buffer_virtual_address(buf),
		.mode = mode,
		.sampleCount = sampleCount,
		.seedSalt = 0u,
		.wi = wi,
		.alpha = alpha,
		.aniso = aniso,
	};
	static constexpr u32 kLocalSize = 256;
	test::dispatch(pl, push, (kBinCount + kLocalSize - 1) / kLocalSize);

	auto out = test::readback<f32>(buf, 0, kBinCount);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

struct ChiSquare {
	f64 statistic;
	f64 dof;
	f64 pValue;
	u64 observedTotal;
	f64 expectedTotal;
	f64 pooledObserved;
	f64 pooledExpected;
};

// pearson chi-square with the standard small-expectation pooling: bins with
// E < 5 are folded into one pooled cell. p-value via the wilson-hilferty
// cube-root normal approximation (accurate to ~1e-3 in p for dof > 10,
// plenty for a pass/fail threshold of 1e-4).
ChiSquare chi_square(
	std::vector<u32> const & observed,
	std::vector<f32> const & expected
) {
	f64 statistic = 0.0;
	u32 cells = 0u;
	f64 pooledO = 0.0;
	f64 pooledE = 0.0;
	u64 observedTotal = 0u;
	f64 expectedTotal = 0.0;
	for (u32 i = 0; i < kBinCount; ++i) {
		observedTotal += observed[i];
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
	// the pooled (E < 5) cell is not folded into the statistic: its
	// per-bin normal approximation is invalid at such low expectations,
	// and the caller bounds pooledObserved against pooledExpected
	// explicitly instead (that bound is what detects a sampler emitting
	// where the pdf claims ~no density)
	f64 const dof = std::fmax((f64)cells - 1.0, 1.0);
	f64 const t = std::cbrt(statistic / dof);
	f64 const mu = 1.0 - 2.0 / (9.0 * dof);
	f64 const sigma = std::sqrt(2.0 / (9.0 * dof));
	f64 const z = (t - mu) / sigma;
	f64 const pValue = 0.5 * std::erfc(z / std::sqrt(2.0));
	return {
		statistic, dof, pValue, observedTotal, expectedTotal, pooledO, pooledE
	};
}

// full sampler-vs-pdf statistical check for one transmission configuration
void check_chi_square(
	u32 const mode,
	f32v3 const wi,
	f32v2 const alpha,
	f32v2 const aniso,
	u32 const seedSalt,
	char const * const residualPngPath = nullptr
) {
	constexpr u32 kSampleCount = 1u << 20;
	auto const observed = (
		run_histogram(mode, wi, alpha, aniso, kSampleCount, seedSalt)
	);
	auto const expected = run_expected(mode, wi, alpha, aniso, kSampleCount);

	CAPTURE(mode);
	CAPTURE(wi.z);
	CAPTURE(alpha.x);
	CAPTURE(alpha.y);
	CAPTURE(aniso.x);
	CAPTURE(aniso.y);

	// no sample may be non-finite or non-unit
	CHECK(observed[kBinCount] == 0u);

	auto const cs = chi_square(observed, expected);
	CAPTURE(cs.statistic);
	CAPTURE(cs.dof);
	CAPTURE(cs.pooledObserved);
	CAPTURE(cs.pooledExpected);
	// every sample must land in some bin
	CHECK(cs.observedTotal == kSampleCount);
	// the expected side (refract-image-restricted pdf plus tir-restricted
	// reflect density) must integrate to ~1 over the sphere: the sampler
	// always produces some direction, and the support-tested pdf owns all
	// of them (2% quadrature slack)
	CHECK(
		cs.expectedTotal
		== doctest::Approx((f64)kSampleCount).epsilon(0.02)
	);
	// sampler and pdf describe the same distribution; the threshold is
	// loose enough (1e-4) that a correct pairing essentially never trips
	// it, while a real mismatch produces p ~ 0
	CHECK(cs.pValue > 1e-4);
	// the pooled (E < 5) cell is excluded from the p-value and bounded
	// separately, same allowance convention as the ggx bounded chi-square:
	// the refraction-cone / tir-threshold support boundaries cut through
	// bin interiors and flicker a handful of samples across them. the
	// exiting mode gets extra slack: it has both the refraction-cone edge
	// AND the tir-threshold edge as separate flicker sources, while
	// entering only has the former (never tir's)
	f64 const pooledAllowance = (
		mode == kModeTransmissionExit ? 160.0 : 64.0
	);
	CHECK(cs.pooledObserved <= cs.pooledExpected + pooledAllowance);

	if (residualPngPath != nullptr) {
		// standardized residual (O - E) / sqrt(max(E, 1)), mapped to a
		// +-4 sigma diverging red/blue image like cull's bsdf-verify view
		std::vector<f32> r(kBinCount), g(kBinCount, 0.0f), b(kBinCount);
		for (u32 i = 0; i < kBinCount; ++i) {
			f32 const e = expected[i];
			f32 const res = (
				((f32)observed[i] - e) / std::sqrt(std::fmax(e, 1.0f))
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

// transmission_integrate.comp: 6 floats/config in, 8 out
struct IntegrateResult {
	f32 rawPdf;
	f32 restrictedPdf;
	f32 tirRestricted;
	f32 tirLib;
	f32 rawFCos;
	f32 restrictedFCos;
};

std::vector<IntegrateResult> run_integrate(
	std::vector<f32> const & configsFlat,
	u32 const thetaSteps,
	u32 const phiSteps
) {
	u32 const count = (u32)configsFlat.size() / 6u;
	REQUIRE(configsFlat.size() == (size_t)count * 6u);
	static constexpr u32 kLocalSize = 64;

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "transmission_integrate.comp",
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
		.byteCount = (u64)count * 8u * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push {
		u64 inVa;
		u64 outVa;
		u32 thetaSteps;
		u32 phiSteps;
		u32 count;
	};
	Push const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.thetaSteps = thetaSteps,
		.phiSteps = phiSteps,
		.count = count,
	};
	test::dispatch(pl, push, (count + kLocalSize - 1) / kLocalSize);

	auto flat = test::readback<f32>(outBuf, 0, count * 8u);
	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);

	std::vector<IntegrateResult> out(count);
	for (u32 i = 0; i < count; ++i) {
		out[i] = {
			flat[i*8+0], flat[i*8+1], flat[i*8+2],
			flat[i*8+3], flat[i*8+4], flat[i*8+5],
		};
	}
	return out;
}

void push_integrate_config(
	std::vector<f32> & flat,
	f32v3 const wi, f32 const ior, f32 const roughness,
	f32 const isInsideMedium
) {
	flat.push_back(wi.x); flat.push_back(wi.y); flat.push_back(wi.z);
	flat.push_back(ior);
	flat.push_back(roughness);
	flat.push_back(isInsideMedium);
}

// layout must match the scalar push block in transmission_sample_sweep.comp
struct SweepPush {
	u64 outVa;
	u32 width;
	u32 height;
	f32 ior;
	f32 isInsideMedium;
};
static_assert(sizeof(SweepPush) == 24);

// 4 floats per texel: nonFiniteCount, maxUnitLengthError, minOwnBranchPdf,
// tirFraction
std::vector<f32> run_sweep(
	u32 const width,
	u32 const height,
	f32 const ior,
	f32 const isInsideMedium
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "transmission_sample_sweep.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto buf = vkof::buffer_create({
		.byteCount = (u64)width * height * 4u * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	SweepPush const push {
		.outVa = vkof::buffer_virtual_address(buf),
		.width = width,
		.height = height,
		.ior = ior,
		.isInsideMedium = isInsideMedium,
	};
	test::dispatch(pl, push, (width + 7u) / 8u, (height + 7u) / 8u);

	auto out = test::readback<f32>(buf, 0, width * height * 4u);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

f32v3 dir_from(f32 const cosTheta, f32 const phi) {
	f32 const sinTheta = (
		std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta))
	);
	return { sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta };
}

constexpr f64 kPi = 3.14159265358979323846;

// f64 dielectric fresnel for the transmittance predictions (same math as
// the lib, snell's-law form)
f64 ref_fresnel_dielectric(f64 const cosTheta, f64 const eta) {
	f64 const cosTheta2 = cosTheta * cosTheta;
	f64 const sinThetaT2 = (1.0 - cosTheta2) / (eta * eta);
	if (sinThetaT2 > 1.0) {
		return 1.0;
	}
	f64 const cosThetaT = std::sqrt(1.0 - sinThetaT2);
	f64 const rParallel = (
		(eta * cosTheta - cosThetaT) / (eta * cosTheta + cosThetaT)
	);
	f64 const rPerspective = (
		(cosTheta - eta * cosThetaT) / (cosTheta + eta * cosThetaT)
	);
	return 0.5 * (rParallel * rParallel + rPerspective * rPerspective);
}

} // namespace

TEST_SUITE("[headless]") {

// -----------------------------------------------------------------------------
// pdf sphere-normalization integrals. the restricted integral applies the
// refract-image support test (see transmission_integrate.comp); the raw
// integral evaluates the lib formula verbatim everywhere and its excess
// over the restricted one measures the wrong-branch (mirror-side) density
// the formula claims at directions the sampler can never emit
// -----------------------------------------------------------------------------

TEST_CASE("transmission pdf: restricted integral is 1 over the sphere (entering)") {
	// entering a denser medium never triggers tir, so the refraction pdf
	// alone must own the full unit mass
	std::vector<f32> flat;
	struct Config { f32 ct; f32 ior; f32 rough; f32 tolerance; };
	std::vector<Config> configs;
	std::vector<f32> const cosThetas = { 0.95f, 0.6f, 0.25f, 0.05f };
	std::vector<f32> const iors = { 1.1f, 1.5f, 2.0f };
	std::vector<f32> const roughnesses = { 0.1f, 0.3f, 0.7f, 1.0f };
	for (f32 const ct : cosThetas) {
		// grazing incidence compresses the refraction lobe against the
		// cone edge faster than fixed-grid quadrature resolves; same
		// resolvable-domain convention as the ggx bounded integrals
		bool const grazing = ct < 0.1f;
		for (f32 const ior : iors) {
			for (f32 const rough : roughnesses) {
				configs.push_back({
					ct, ior, rough, grazing ? 0.05f : 0.02f
				});
			}
		}
	}
	for (auto const & c : configs) {
		push_integrate_config(
			flat, dir_from(c.ct, 0.4f), c.ior, c.rough, 0.0f
		);
	}
	auto const out = run_integrate(flat, 400u, 400u);
	f64 maxSpurious = 0.0;
	for (u32 i = 0; i < configs.size(); ++i) {
		CAPTURE(i);
		CAPTURE(configs[i].ct);
		CAPTURE(configs[i].ior);
		CAPTURE(configs[i].rough);
		CHECK(std::isfinite(out[i].restrictedPdf));
		CHECK(
			out[i].restrictedPdf
			== doctest::Approx(1.0).epsilon(configs[i].tolerance)
		);
		// no tir when entering a denser medium
		CHECK(out[i].tirRestricted == 0.0f);
		// the raw formula can only add mass on top of the image set
		CHECK(out[i].rawPdf >= out[i].restrictedPdf - 0.01f);
		maxSpurious = std::max(
			maxSpurious, (f64)out[i].rawPdf - (f64)out[i].restrictedPdf
		);
	}
	// reported finding, quantified: the raw openPbrTransmissionPdf claims
	// wrong-branch density beyond the sampler's image; the mixture pdf in
	// production only overweights with it (mis-safe direction), so this is
	// documented rather than bounded to zero
	MESSAGE(
		"max wrong-branch (raw - restricted) pdf mass, entering: ",
		maxSpurious
	);
}

TEST_CASE("transmission pdf: restricted + tir-restricted integral is 1 (inside medium)") {
	// exiting toward a rarer medium: mass splits between refraction and
	// the tir reflect fallback; together they own every sample
	std::vector<f32> flat;
	struct Config { f32 ct; f32 ior; f32 rough; };
	std::vector<Config> configs;
	std::vector<f32> const cosThetas = { 0.95f, 0.75f, 0.5f, 0.2f };
	std::vector<f32> const iors = { 1.33f, 1.5f, 2.0f };
	std::vector<f32> const roughnesses = { 0.1f, 0.3f, 0.7f, 1.0f };
	for (f32 const ct : cosThetas) {
		for (f32 const ior : iors) {
			for (f32 const rough : roughnesses) {
				configs.push_back({ ct, ior, rough });
			}
		}
	}
	for (auto const & c : configs) {
		push_integrate_config(
			flat, dir_from(c.ct, 0.4f), c.ior, c.rough, 1.0f
		);
	}
	auto const out = run_integrate(flat, 400u, 400u);
	for (u32 i = 0; i < configs.size(); ++i) {
		CAPTURE(i);
		CAPTURE(configs[i].ct);
		CAPTURE(configs[i].ior);
		CAPTURE(configs[i].rough);
		f64 const total = (
			(f64)out[i].restrictedPdf + (f64)out[i].tirRestricted
		);
		CHECK(std::isfinite(out[i].restrictedPdf));
		CHECK(std::isfinite(out[i].tirRestricted));
		CHECK(total == doctest::Approx(1.0).epsilon(0.03));
		// openPbrTransmissionTirReflectPdf now restricts itself to
		// genuinely-TIR h's internally (previously unrestricted --
		// claimed density everywhere, which the mixture-lobe-selection
		// milestone found double-counting against a separately-
		// normalized reflecting lobe when both are summed in
		// openPbrEvaluatePdfSelected). tirLib and tirRestricted are the
		// same quantity by construction now; the meaningful invariant is
		// that restrictedPdf (refraction share) plus tirRestricted
		// (tir-reflect share) together normalize to 1, checked above
		CHECK(out[i].tirLib == doctest::Approx(out[i].tirRestricted).epsilon(1e-4));
	}
}

TEST_CASE("transmission pdf: tir fraction grows toward grazing incidence (inside medium)") {
	// physical sanity on the mass split: past the critical angle most
	// half vectors reflect internally, so the tir share of the sampler's
	// mass must increase monotonically as the incidence goes grazing
	std::vector<f32> flat;
	std::vector<f32> const cosThetas = { 0.95f, 0.8f, 0.66f, 0.5f, 0.3f };
	for (f32 const ct : cosThetas) {
		push_integrate_config(flat, dir_from(ct, 0.0f), 1.5f, 0.3f, 1.0f);
	}
	auto const out = run_integrate(flat, 400u, 400u);
	f32 prev = -1.0f;
	for (u32 i = 0; i < cosThetas.size(); ++i) {
		CAPTURE(cosThetas[i]);
		f32 const tirShare = out[i].tirRestricted;
		MESSAGE("ct=", cosThetas[i], " tir share: ", tirShare);
		CHECK(tirShare >= prev - 0.01f);
		prev = tirShare;
	}
	// at ct = 0.3 (70+ degrees, far past the 41.8-degree critical angle)
	// tir must dominate
	CHECK(out[cosThetas.size() - 1u].tirRestricted > 0.5f);
}

// -----------------------------------------------------------------------------
// chi-square: the sampler and the (support-tested) pdf describe the same
// distribution
// -----------------------------------------------------------------------------

TEST_CASE("transmission sampler vs pdf: chi-square (entering)") {
	// roughness floor 0.2: below that the refraction lobe concentrates
	// into too few bins at this grid resolution, same resolvability
	// convention as the ggx bounded chi-square
	u32 salt = 400u;
	std::vector<f32> const roughnesses = { 0.2f, 0.5f, 1.0f };
	std::vector<f32> const cosThetas = { 1.0f, 0.7f, 0.3f };
	for (f32 const rough : roughnesses) {
		for (f32 const ct : cosThetas) {
			f32 const st = std::sqrt(std::max(0.0f, 1.0f - ct * ct));
			check_chi_square(
				kModeTransmissionEnter,
				{ st, 0.0f, ct }, { rough, 1.5f }, { 0.0f, 0.0f }, salt++
			);
		}
	}
	// dense flint corner
	check_chi_square(
		kModeTransmissionEnter,
		{ 0.866f, 0.0f, 0.5f }, { 0.5f, 2.0f }, { 0.0f, 0.0f }, salt++
	);
}

TEST_CASE("transmission sampler vs pdf: chi-square (inside medium, tir split)") {
	// spans the critical-angle transition where the sampler's mass splits
	// between the refraction pdf and the tir reflect fallback
	u32 salt = 500u;
	std::vector<f32> const roughnesses = { 0.2f, 0.5f, 1.0f };
	std::vector<f32> const cosThetas = { 0.95f, 0.7f, 0.4f, 0.15f };
	for (f32 const rough : roughnesses) {
		for (f32 const ct : cosThetas) {
			f32 const st = std::sqrt(std::max(0.0f, 1.0f - ct * ct));
			check_chi_square(
				kModeTransmissionExit,
				{ st, 0.0f, ct }, { rough, 1.5f }, { 0.0f, 0.0f }, salt++
			);
		}
	}
}

TEST_CASE("transmission sampler vs pdf: chi-square (anisotropic)") {
	// the aniso D / smith G1 / vndf sampler triple has no external
	// reference to check against (materialx's smith is scalar-alpha, and
	// no surveyed renderer keeps tir inside the transmission lobe), so
	// this is the primary instrument for the anisotropic path. an
	// off-axis wi and a rotated frame are both needed: an in-plane wi
	// with rotation 0 cannot tell alpha.x from alpha.y
	u32 salt = 700u;
	// anisotropy 0 is the control: it isolates whether a failure comes from
	// the aniso axes or from the off-plane wi these configs also introduce
	std::vector<f32> const anisotropies = { 0.0f, 0.4f, 0.8f };
	std::vector<f32> const rotations = { 0.0f, 0.7f };
	std::vector<f32v3> const wis = {
		{ 0.0f, 0.0f, 1.0f },
		{ 0.5f, 0.4f, 0.768f },
		{ 0.62f, 0.5f, 0.605f },
	};
	for (f32 const anisotropy : anisotropies) {
		for (f32 const rotation : rotations) {
			for (f32v3 const & wi : wis) {
				check_chi_square(
					kModeTransmissionEnter,
					wi, { 0.5f, 1.5f }, { anisotropy, rotation }, salt++
				);
				check_chi_square(
					kModeTransmissionExit,
					wi, { 0.5f, 1.5f }, { anisotropy, rotation }, salt++
				);
			}
		}
	}
}

TEST_CASE("transmission sampler vs pdf: chi-square residual heatmaps") {
	check_chi_square(
		kModeTransmissionEnter, { 0.714f, 0.0f, 0.7f }, { 0.5f, 1.5f }, { 0.0f, 0.0f }, 61u,
		TRANSMISSION_OUTPUT_DIR "transmission_chi2_residual_enter.png"
	);
	check_chi_square(
		kModeTransmissionExit, { 0.714f, 0.0f, 0.7f }, { 0.5f, 1.5f }, { 0.0f, 0.0f }, 62u,
		TRANSMISSION_OUTPUT_DIR "transmission_chi2_residual_exit.png"
	);
}

// -----------------------------------------------------------------------------
// single-scatter transmittance furnace: integral of f |cos| over the
// transmitted side. no energy compensation exists on this lobe (unlike the
// reflection lobes' turquin/kulla-conty terms), so the identity is an
// upper bound plus the smooth-limit prediction, not == 1
// -----------------------------------------------------------------------------

TEST_CASE("transmission furnace: transmittance bounded by 1 and approaches 1 - F at low roughness") {
	std::vector<f32> flat;
	struct Config { f32 ct; f32 ior; f32 rough; bool inside; };
	std::vector<Config> configs;
	std::vector<f32> const cosThetas = { 0.95f, 0.7f, 0.4f };
	std::vector<f32> const iors = { 1.1f, 1.5f, 2.0f };
	std::vector<f32> const roughnesses = { 0.05f, 0.3f, 0.7f, 1.0f };
	for (f32 const ct : cosThetas) {
		for (f32 const ior : iors) {
			for (f32 const rough : roughnesses) {
				configs.push_back({ ct, ior, rough, false });
			}
		}
	}
	for (auto const & c : configs) {
		push_integrate_config(
			flat, dir_from(c.ct, 0.0f), c.ior, c.rough, c.inside ? 1.0f : 0.0f
		);
	}
	auto const out = run_integrate(flat, 400u, 400u);
	for (u32 i = 0; i < configs.size(); ++i) {
		CAPTURE(configs[i].ct);
		CAPTURE(configs[i].ior);
		CAPTURE(configs[i].rough);
		f32 const transmittance = out[i].restrictedFCos;
		CHECK(std::isfinite(transmittance));
		CHECK(transmittance >= 0.0f);
		// single scattering can only lose energy against the smooth-limit
		// transmittance; roughness 0.05 is a genuine near-delta lobe this
		// 400x400 fixed grid can't resolve (measured 1.04 here, converging
		// to the correct ~0.999 only past 800x800) -- same fixed-grid
		// resolvability convention as the metallic-lobe furnace test,
		// which excludes roughness < 0.1 from its strict bound for the
		// identical reason
		f32 const tolerance = configs[i].rough <= 0.05f ? 1.05f : 1.005f;
		CHECK(transmittance <= tolerance);
		if (configs[i].rough <= 0.05f) {
			// smooth limit: T -> 1 - F(muI); widened epsilon covers the
			// same under-resolved near-delta quadrature error as the
			// bound above
			f64 const fres = ref_fresnel_dielectric(
				(f64)configs[i].ct, (f64)configs[i].ior
			);
			CHECK(
				transmittance
				== doctest::Approx(1.0 - fres).epsilon(0.06)
			);
		}
	}
}

TEST_CASE("transmission furnace: single-scatter energy loss grows with roughness") {
	std::vector<f32> flat;
	std::vector<f32> const roughnesses = { 0.1f, 0.3f, 0.6f, 1.0f };
	for (f32 const rough : roughnesses) {
		push_integrate_config(flat, dir_from(0.8f, 0.0f), 1.5f, rough, 0.0f);
	}
	auto const out = run_integrate(flat, 400u, 400u);
	f32 prev = 2.0f;
	for (u32 i = 0; i < roughnesses.size(); ++i) {
		CAPTURE(roughnesses[i]);
		MESSAGE(
			"rough=", roughnesses[i],
			" transmittance: ", out[i].restrictedFCos
		);
		CHECK(out[i].restrictedFCos < prev + 0.005f);
		prev = out[i].restrictedFCos;
	}
}

TEST_CASE("transmission furnace: transmittance heatmap (incidence x roughness)") {
	// grayscale transmittance map, entering ior 1.5 glass; red would mean
	// NaN. the strict per-config assertions above cover the resolvable
	// domain, the image shows the whole square honestly
	constexpr u32 kWidth = 48u;
	constexpr u32 kHeight = 48u;
	std::vector<f32> flat;
	for (u32 y = 0; y < kHeight; ++y) {
		f32 const rough = 0.02f + 0.98f * (f32)y / (f32)(kHeight - 1);
		for (u32 x = 0; x < kWidth; ++x) {
			f32 const ct = 1.0f - 0.97f * (f32)x / (f32)(kWidth - 1);
			push_integrate_config(flat, dir_from(ct, 0.0f), 1.5f, rough, 0.0f);
		}
	}
	auto const out = run_integrate(flat, 200u, 200u);
	std::vector<f32> img((size_t)kWidth * kHeight);
	f64 wrongBranchWorst = 0.0;
	for (u32 i = 0; i < (u32)(kWidth * kHeight); ++i) {
		img[i] = (
			std::isfinite(out[i].restrictedFCos)
			? out[i].restrictedFCos
			: std::nanf("")
		);
		if (
			std::isfinite(out[i].rawFCos)
			&& std::isfinite(out[i].restrictedFCos)
		) {
			wrongBranchWorst = std::max(
				wrongBranchWorst,
				(f64)out[i].rawFCos - (f64)out[i].restrictedFCos
			);
		}
	}
	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		img, img, img, kWidth, kHeight,
		TRANSMISSION_OUTPUT_DIR "transmission_furnace_heatmap.png",
		&nanCount
	);
	CHECK(ok);
	CHECK(nanCount == 0);
	// reported finding, quantified over the sweep: energy the raw
	// evaluateF formula claims at wrong-branch directions the sampler
	// can never emit (a light-sampling overestimate hazard, not a
	// sampling one)
	MESSAGE(
		"worst wrong-branch f|cos| mass over (incidence x roughness): ",
		wrongBranchWorst
	);
}

// -----------------------------------------------------------------------------
// NaN sweeps
// -----------------------------------------------------------------------------

TEST_CASE("transmission sampler: NaN sweep heatmaps (entering, exiting, matched ior)") {
	// dense (incidence x roughness) sweep of the production sample path,
	// 8x8 xi grid per texel including the exact 0 stratum. wi covers the
	// full [-1, 1] cosine range; assertions apply to the production
	// domain (wi above the horizon), the below-horizon half is documented
	// by the image only. the matched-ior config used to document a known
	// ior = 1 pdf degeneracy (fixed 2026-07-15: etaI == etaT now takes the
	// thin-walled delta lobe, pdf == 1 exactly), so all four configs are
	// expected clean here
	constexpr u32 kSize = 256u;
	struct SweepConfig {
		f32 ior;
		f32 inside;
		char const * path;
	};
	std::vector<SweepConfig> const configs = {
		{
			1.5f, 0.0f,
			TRANSMISSION_OUTPUT_DIR "transmission_nan_sweep_enter.png",
		},
		{
			1.5f, 1.0f,
			TRANSMISSION_OUTPUT_DIR "transmission_nan_sweep_exit.png",
		},
		{
			2.42f, 1.0f,
			TRANSMISSION_OUTPUT_DIR "transmission_nan_sweep_exit_diamond.png",
		},
		{
			1.0f, 0.0f,
			TRANSMISSION_OUTPUT_DIR "transmission_nan_sweep_matched_ior.png",
		},
	};
	for (auto const & cfg : configs) {
		auto const out = run_sweep(kSize, kSize, cfg.ior, cfg.inside);

		std::vector<f32> r((size_t)kSize * kSize);
		std::vector<f32> g((size_t)kSize * kSize);
		std::vector<f32> b((size_t)kSize * kSize);
		u32 nonFiniteProduction = 0u;
		u32 nonFiniteBelow = 0u;
		u32 pdfZeroProduction = 0u;
		f32 maxLenErr = 0.0f;
		for (u32 y = 0; y < kSize; ++y) {
			for (u32 x = 0; x < kSize; ++x) {
				u32 const i = y * kSize + x;
				// x maps to cosThetaI in [-1, 1]; the above-horizon
				// production half starts at the midpoint
				bool const aboveHorizon = x > kSize / 2u;
				if (out[i*4+0] > 0.0f) {
					if (aboveHorizon) { nonFiniteProduction++; }
					else { nonFiniteBelow++; }
					r[i] = std::nanf("");
					g[i] = std::nanf("");
					b[i] = std::nanf("");
					continue;
				}
				if (aboveHorizon) {
					maxLenErr = std::fmax(maxLenErr, out[i*4+1]);
					if (!(out[i*4+2] > 0.0f)) { pdfZeroProduction++; }
				}
				// base image: tir fraction in red, tonemapped min own-pdf
				// in green/blue
				f32 const v = out[i*4+2] / (out[i*4+2] + 1.0f);
				r[i] = out[i*4+3];
				g[i] = v;
				b[i] = v;
			}
		}
		u32 nanCount = 0;
		bool const ok = test::write_heatmap_png(
			r, g, b, kSize, kSize, cfg.path, &nanCount
		);
		CHECK(ok);
		CAPTURE(cfg.ior);
		CAPTURE(cfg.inside);
		MESSAGE(
			"sampler sweep ior=", cfg.ior, " inside=", cfg.inside,
			": nonFinite production/below-horizon = ",
			nonFiniteProduction, "/", nonFiniteBelow,
			", pdfZero production = ", pdfZeroProduction,
			", maxLenErr = ", maxLenErr
		);
		CHECK(nonFiniteProduction == 0u);
		CHECK(maxLenErr < 1e-3f);
		// pdf must be strictly positive at the sampler's own outputs on
		// the production domain, or mis weights divide by zero
		CHECK(pdfZeroProduction == 0u);
	}
}

TEST_CASE("transmission sampler: tir fraction is zero when entering a denser medium") {
	// glsl refract with eta < 1 always succeeds, so the entering sweep
	// must never take the tir fallback anywhere in the parameter square
	constexpr u32 kSize = 128u;
	auto const out = run_sweep(kSize, kSize, 1.5f, 0.0f);
	u32 tirTexels = 0u;
	for (u32 i = 0; i < kSize * kSize; ++i) {
		if (out[i*4+3] > 0.0f) { tirTexels++; }
	}
	CHECK(tirTexels == 0u);
}

} // TEST_SUITE("[headless]")
