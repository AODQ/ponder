#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <ponder/energy-tables.hpp>
#include <ponder/environment-tables.hpp>
#include <ponder/zeltner-tables.hpp>
#include "util.hpp"

#include <tinyexr.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// env-map next-event-estimation / MIS unbiasedness.
//
// test-environment-sampling.cpp already proves the *sampler* and the *pdf*
// describe the same distribution. That is necessary but not sufficient: the
// path tracer combines that sampler with openPbrSampleWo through the balance
// heuristic, and a correct pdf pair can still lose energy if either half of
// the combined estimator is mis-weighted or silently drops its contribution.
// Nothing in the suite tested the combination until this file.
//
// The estimator under test is the direct-lighting integral at one surface
// vertex,
//
//   \int_{S^2} L(\omega) f(\omega_i, \omega) |\cos\theta| d\omega
//
// run three ways over the same material and map (env_direct_lighting.comp:
// NEE-only, BSDF-only, MIS) and compared against a deterministic quadrature
// (env_direct_lighting_reference.comp). All four must coincide. There is no
// geometry and there are no shadow rays, so a failure here is in the
// sampling/pdf/MIS arithmetic and nothing else.
//
// Why this shape, rather than a rendered scene: where the map is locally very
// bright, envPdf >> bsdfPdf, so the BSDF half's balance weight collapses to
// ~0 and essentially the entire contribution has to arrive through the NEE
// half. Any under-delivery there reads as energy loss *proportional to
// envPdf* -- invisible on a smooth map, and increasingly severe the brighter
// and more compact the feature. That is the signature of a rough metal or
// clearcoat going dark on the side facing a sun, so the cases below sweep
// exactly that axis: sun contrast, and surface roughness (which sets how
// small bsdfPdf is next to the sun's envPdf).
//
// What it found, as of 2026-08-03. The MIS arithmetic is clean -- every
// configuration on the moderate maps agrees with quadrature to well inside
// its own error bar, so the balance weights are not losing energy. Two
// separate defects underneath it are not clean, and they are independent:
//
//   1. Bilinear radiance against a texel-exact pdf. environmentMapPdf reads
//      one texel with texelFetch; environmentMapRadiance reads four with
//      texture(). A texel beside a sun therefore still returns part of that
//      sun while being priced at its own sky density, which is a rare and
//      enormous sample. Formally unbiased -- the tail carries the energy back
//      -- but resolve.comp's fireflyClampLuminance deletes exactly that tail,
//      and it defaults to on. Measured: a clamp an order of magnitude above
//      the estimator's own mean costs NEE 41% and MIS 31% of the direct
//      lighting on a 2x2-texel sun, and 0.00% on a smooth map. Switching the
//      fetch to nearest takes every one of those numbers to 0.00%, which is
//      what attributes the loss to this and not to defect 2.
//
//   2. Sub-texel jitter collapse. environmentMapImportanceSample derives the
//      continuous position inside the picked texel from the *leftover
//      mantissa bits* of the same random number that picked it, which stop
//      carrying information as the texel count approaches 2^24. Those samples
//      land exactly on their texel's edge, and the dir -> uv round trip every
//      consumer performs to recover the pdf then resolves a measurable
//      fraction of them onto the neighbouring texel -- so the density the
//      sampler actually has and the density it reports come apart. This one
//      is a real bias, not a tail: NEE reads +6.4% high on a 2x2-texel sun
//      and the number does not move when the sample count goes up 32x, while
//      BSDF-only stays clean on the same map. It scales with resolution, and
//      a 4096x2048 HDRI is the bad end of that scale.
// ---------------------------------------------------------------------------

namespace {

// mirrors util-environment-map.glsl's EnvironmentMapHandles field order
struct GpuEnvironmentMapHandles {
	u32 radiance;
	u32 pdf;
	u32 aliasProb;
	u32 aliasIndex;
};

GpuEnvironmentMapHandles handles_from_tables(
	ponder::EnvironmentTables const & tables
) {
	return {
		.radiance = tables.radianceHandle,
		.pdf = tables.pdfHandle,
		.aliasProb = tables.aliasProbHandle,
		.aliasIndex = tables.aliasIndexHandle,
	};
}

// one long-lived handle pair for the whole file. test-mixture-lobe-selection
// .cpp documents why these are not created and destroyed per TEST_CASE: the
// bindless-slot churn made unrelated cases fail depending on which other
// cases ran. leaked for the lifetime of the test binary, deliberately
u32 shared_zeltner_handle() {
	static ponder::ZeltnerTables const tables = ponder::zeltner_tables_create();
	return tables.zeltnerLtcParamHandle;
}

u32 shared_kulla_conty_handle() {
	static ponder::EnergyTables const tables = ponder::energy_tables_create();
	return tables.kullaContyEnergyHandle;
}

// writes an rgba float exr and loads it back through the production loader,
// so every case exercises environment_tables_create end to end -- same
// approach as test-environment-sampling.cpp
ponder::EnvironmentTables tables_from_pixels(
	char const * const path,
	i32 const width,
	i32 const height,
	std::vector<f32> const & pixels
) {
	REQUIRE(pixels.size() == (size_t)width * height * 4);
	char const * err = nullptr;
	i32 const ret = (
		SaveEXR(pixels.data(), width, height, 4, /*save_as_fp16=*/0, path, &err)
	);
	if (ret != TINYEXR_SUCCESS) {
		MESSAGE("SaveEXR failed: ", err ? err : "unknown error");
		if (err) {
			FreeEXRErrorMessage(err);
		}
	}
	REQUIRE(ret == TINYEXR_SUCCESS);
	return ponder::environment_tables_create(path);
}

void set_pixel(
	std::vector<f32> & pixels,
	i32 const width,
	i32 const x,
	i32 const y,
	f32 const r,
	f32 const g,
	f32 const b
) {
	size_t const i = ((size_t)y * width + x) * 4;
	pixels[i + 0] = r;
	pixels[i + 1] = g;
	pixels[i + 2] = b;
	pixels[i + 3] = 1.0f;
}

constexpr i32 kMapWidth = 128;
constexpr i32 kMapHeight = 64;

// a "sky" that is smooth and everywhere positive (so no row collapses into
// the builder's uniform fallback) plus an optional square sun of
// `sunRadiance` and `sunHalfExtent` texels, centred well above the horizon
// (v ~ 0.25, i.e. theta ~ 45 degrees) so it sits in the upper hemisphere of
// the y-up shading frame the estimator shaders use.
//
// sunHalfExtent == 0 builds the sunless control map
ponder::EnvironmentTables make_sky(
	char const * const path,
	f32 const sunRadiance,
	i32 const sunHalfExtent
) {
	std::vector<f32> pixels((size_t)kMapWidth * kMapHeight * 4);
	i32 const sunX = kMapWidth * 3 / 4;
	i32 const sunY = kMapHeight / 4;
	for (i32 y = 0; y < kMapHeight; ++y) {
		for (i32 x = 0; x < kMapWidth; ++x) {
			// mildly coloured and mildly varying, so a channel swap or a
			// dropped gradient cannot hide behind a flat grey
			f32 const t = (f32)x / (f32)kMapWidth;
			f32 const s = (f32)y / (f32)kMapHeight;
			f32 const r = 0.08f + 0.05f * t;
			f32 const g = 0.10f + 0.04f * s;
			f32 const b = 0.14f + 0.03f * (1.0f - t);
			bool const isSun = (
				sunHalfExtent > 0
				&& std::abs(x - sunX) < sunHalfExtent
				&& std::abs(y - sunY) < sunHalfExtent
			);
			if (isSun) {
				set_pixel(
					pixels, kMapWidth, x, y,
					sunRadiance, sunRadiance * 0.95f, sunRadiance * 0.85f
				);
			} else {
				set_pixel(pixels, kMapWidth, x, y, r, g, b);
			}
		}
	}
	return tables_from_pixels(path, kMapWidth, kMapHeight, pixels);
}

// ---------------------------------------------------------------------------
// -- gpu push layouts (scalar block layout; field order must match the .comp)
// ---------------------------------------------------------------------------

struct EnvDlPush {
	u64 sumVa;
	u64 sumSqVa;
	u64 badVa;
	GpuEnvironmentMapHandles handles;
	u32 kullaContyEnergyHandle;
	u32 zeltnerLtcParamHandle;
	u32 mode;
	u32 threadCount;
	u32 samplesPerThread;
	u32 seedSalt;
	f32 rotation;
	f32 wi[3];
	f32 baseColor[3];
	f32 baseMetalness;
	f32 specularRoughness;
	f32 specularIor;
	f32 coatWeight;
	f32 coatRoughness;
	f32 coatIor;
	f32 fireflyClampLuminance;
	u32 radianceNearest;
};

struct EnvRefPush {
	u64 sumVa;
	GpuEnvironmentMapHandles handles;
	u32 kullaContyEnergyHandle;
	u32 zeltnerLtcParamHandle;
	u32 subGrid;
	f32 rotation;
	f32 wi[3];
	f32 baseColor[3];
	f32 baseMetalness;
	f32 specularRoughness;
	f32 specularIor;
	f32 coatWeight;
	f32 coatRoughness;
	f32 coatIor;
};

struct EnvJitterPush {
	u64 binVa;
	GpuEnvironmentMapHandles handles;
	u32 sampleCount;
	u32 binCount;
	u32 seedSalt;
	f32 rotation;
};

// ---------------------------------------------------------------------------
// -- material configurations under test
// ---------------------------------------------------------------------------

struct Material {
	char const * name;
	f32 baseColor[3];
	f32 baseMetalness;
	f32 specularRoughness;
	f32 specularIor;
	f32 coatWeight;
	f32 coatRoughness;
	f32 coatIor;
};

Material make_metal(char const * const name, f32 const roughness) {
	return Material {
		.name = name,
		.baseColor = { 0.95f, 0.78f, 0.42f },
		.baseMetalness = 1.0f,
		.specularRoughness = roughness,
		.specularIor = 1.5f,
		.coatWeight = 0.0f,
		.coatRoughness = 0.1f,
		.coatIor = 1.5f,
	};
}

Material make_coated_diffuse(char const * const name, f32 const coatRoughness) {
	return Material {
		.name = name,
		.baseColor = { 0.18f, 0.32f, 0.55f },
		.baseMetalness = 0.0f,
		.specularRoughness = 0.4f,
		.specularIor = 1.5f,
		.coatWeight = 1.0f,
		.coatRoughness = coatRoughness,
		.coatIor = 1.5f,
	};
}

Material make_diffuse(char const * const name) {
	return Material {
		.name = name,
		.baseColor = { 0.6f, 0.55f, 0.5f },
		.baseMetalness = 0.0f,
		.specularRoughness = 1.0f,
		.specularIor = 1.5f,
		.coatWeight = 0.0f,
		.coatRoughness = 0.1f,
		.coatIor = 1.5f,
	};
}

// ---------------------------------------------------------------------------
// -- dispatch helpers
// ---------------------------------------------------------------------------

struct Rgb {
	f64 r;
	f64 g;
	f64 b;
};

f64 luminance(Rgb const & c) {
	return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

// relative difference in luminance, symmetric-free: measured against the
// reference, since that is the quantity "N% too dark" refers to
f64 relative_error(Rgb const & measured, Rgb const & reference) {
	f64 const ref = luminance(reference);
	if (ref <= 0.0) {
		return luminance(measured) <= 0.0 ? 0.0 : 1.0;
	}
	return (luminance(measured) - ref) / ref;
}

struct EstimatorResult {
	Rgb mean;
	// per-sample luminance standard deviation; MIS is expected to beat the
	// single-technique estimator that is bad at this map
	f64 luminanceStdDev;
	// standard error of the luminance mean, as a fraction of that mean.
	// every tolerance below is expressed against this rather than against a
	// guessed percentage: a hard sun makes the single-technique estimators
	// genuinely noisy, and a fixed threshold cannot tell "under-converged"
	// apart from "biased" -- which is the entire question this file exists
	// to answer.
	//
	// derived from the spread of independent replicates, not from the
	// per-sample sample variance. these estimators are heavy-tailed on a
	// high-contrast map (env importance sampling occasionally draws a dim
	// texel whose bilinear radiance still carries part of the sun, so a
	// large value arrives at a small pdf), and a variance measured from the
	// samples of a single run systematically under-reports the tail that run
	// has not yet seen. replicate spread does not have that failure mode
	f64 relativeStandardError;
	u32 nonFiniteCount;
};

constexpr u32 kThreadCount = 8192u;
constexpr u32 kLocalSize = 64u;
// independent runs whose spread becomes the error bar; each run gets
// samplesPerThread / kReplicateCount samples, so the total work is unchanged
constexpr u32 kReplicateCount = 16u;

EstimatorResult run_estimator(
	GpuEnvironmentMapHandles const & handles,
	Material const & material,
	f32 const (& wi)[3],
	f32 const rotation,
	u32 const mode,
	u32 const samplesPerThread,
	u32 const seedSalt,
	f32 const fireflyClampLuminance = 0.0f,
	bool const radianceNearest = false
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR, TEST_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "slang/entry/env_direct_lighting.slang",
		.includePaths = srat::slice<char const * const>(includePaths, 2),
	});
	REQUIRE(pl.id != 0);

	u32 const floatCount = kThreadCount * 4u;
	auto sumBuf = vkof::buffer_create({
		.byteCount = floatCount * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	auto sumSqBuf = vkof::buffer_create({
		.byteCount = floatCount * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	auto badBuf = test::make_buffer_u32(1u);
	u32 const zero = 0u;
	vkof::buffer_upload({
		.buffer = badBuf,
		.byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(&zero), sizeof(u32)
		),
	});

	u32 const perReplicate = std::max(samplesPerThread / kReplicateCount, 1u);
	std::vector<Rgb> replicateMeans;
	replicateMeans.reserve(kReplicateCount);
	f64 totalLuminance = 0.0;
	f64 totalSqLuminance = 0.0;
	for (u32 replicate = 0; replicate < kReplicateCount; ++replicate) {
		EnvDlPush const push {
			.sumVa = vkof::buffer_virtual_address(sumBuf),
			.sumSqVa = vkof::buffer_virtual_address(sumSqBuf),
			.badVa = vkof::buffer_virtual_address(badBuf),
			.handles = handles,
			.kullaContyEnergyHandle = shared_kulla_conty_handle(),
			.zeltnerLtcParamHandle = shared_zeltner_handle(),
			.mode = mode,
			.threadCount = kThreadCount,
			.samplesPerThread = perReplicate,
			.seedSalt = seedSalt * 1024u + replicate,
			.rotation = rotation,
			.wi = { wi[0], wi[1], wi[2] },
			.baseColor = {
				material.baseColor[0],
				material.baseColor[1],
				material.baseColor[2],
			},
			.baseMetalness = material.baseMetalness,
			.specularRoughness = material.specularRoughness,
			.specularIor = material.specularIor,
			.coatWeight = material.coatWeight,
			.coatRoughness = material.coatRoughness,
			.coatIor = material.coatIor,
			.fireflyClampLuminance = fireflyClampLuminance,
			.radianceNearest = radianceNearest ? 1u : 0u,
		};
		test::dispatch(pl, push, (kThreadCount + kLocalSize - 1u) / kLocalSize);

		auto const sums = test::readback<f32>(sumBuf, 0, floatCount);
		auto const sumSqs = test::readback<f32>(sumSqBuf, 0, floatCount);
		f64 totalR = 0.0;
		f64 totalG = 0.0;
		f64 totalB = 0.0;
		for (u32 i = 0; i < kThreadCount; ++i) {
			totalR += (f64)sums[i * 4u + 0u];
			totalG += (f64)sums[i * 4u + 1u];
			totalB += (f64)sums[i * 4u + 2u];
			totalLuminance += (f64)sums[i * 4u + 3u];
			totalSqLuminance += (f64)sumSqs[i * 4u + 3u];
		}
		f64 const n = (f64)kThreadCount * (f64)perReplicate;
		replicateMeans.push_back(Rgb { totalR / n, totalG / n, totalB / n });
	}

	auto const bad = test::readback<u32>(badBuf, 0, 1u);
	vkof::buffer_destroy(sumBuf);
	vkof::buffer_destroy(sumSqBuf);
	vkof::buffer_destroy(badBuf);
	vkof::pipeline_destroy(pl);

	Rgb grand { 0.0, 0.0, 0.0 };
	for (Rgb const & replicateMean : replicateMeans) {
		grand.r += replicateMean.r / (f64)kReplicateCount;
		grand.g += replicateMean.g / (f64)kReplicateCount;
		grand.b += replicateMean.b / (f64)kReplicateCount;
	}
	f64 const grandLuminance = luminance(grand);

	f64 spread = 0.0;
	for (Rgb const & replicateMean : replicateMeans) {
		f64 const d = luminance(replicateMean) - grandLuminance;
		spread += d * d;
	}
	f64 const replicateVariance = spread / (f64)(kReplicateCount - 1u);
	f64 const standardError = (
		std::sqrt(replicateVariance / (f64)kReplicateCount)
	);

	f64 const totalSamples = (
		(f64)kThreadCount * (f64)perReplicate * (f64)kReplicateCount
	);
	f64 const perSampleMean = totalLuminance / totalSamples;
	f64 const perSampleVariance = (
		std::fmax(
			totalSqLuminance / totalSamples - perSampleMean * perSampleMean,
			0.0
		)
	);

	return {
		.mean = grand,
		.luminanceStdDev = std::sqrt(perSampleVariance),
		.relativeStandardError = (
			grandLuminance > 0.0 ? standardError / grandLuminance : 0.0
		),
		.nonFiniteCount = bad[0],
	};
}

Rgb run_reference(
	GpuEnvironmentMapHandles const & handles,
	Material const & material,
	f32 const (& wi)[3],
	f32 const rotation,
	u32 const subGrid
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR, TEST_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "slang/entry/env_direct_lighting_reference.slang",
		.includePaths = srat::slice<char const * const>(includePaths, 2),
	});
	REQUIRE(pl.id != 0);

	u32 const texelCount = (u32)kMapWidth * (u32)kMapHeight;
	u32 const floatCount = texelCount * 3u;
	auto sumBuf = vkof::buffer_create({
		.byteCount = floatCount * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	EnvRefPush const push {
		.sumVa = vkof::buffer_virtual_address(sumBuf),
		.handles = handles,
		.kullaContyEnergyHandle = shared_kulla_conty_handle(),
		.zeltnerLtcParamHandle = shared_zeltner_handle(),
		.subGrid = subGrid,
		.rotation = rotation,
		.wi = { wi[0], wi[1], wi[2] },
		.baseColor = {
			material.baseColor[0],
			material.baseColor[1],
			material.baseColor[2],
		},
		.baseMetalness = material.baseMetalness,
		.specularRoughness = material.specularRoughness,
		.specularIor = material.specularIor,
		.coatWeight = material.coatWeight,
		.coatRoughness = material.coatRoughness,
		.coatIor = material.coatIor,
	};
	test::dispatch(pl, push, (texelCount + kLocalSize - 1u) / kLocalSize);

	auto const sums = test::readback<f32>(sumBuf, 0, floatCount);
	vkof::buffer_destroy(sumBuf);
	vkof::pipeline_destroy(pl);

	Rgb total { 0.0, 0.0, 0.0 };
	for (u32 i = 0; i < texelCount; ++i) {
		total.r += (f64)sums[i * 3u + 0u];
		total.g += (f64)sums[i * 3u + 1u];
		total.b += (f64)sums[i * 3u + 2u];
	}
	return total;
}

// one row of the printed comparison table; also what the assertions read
struct Comparison {
	Rgb reference;
	EstimatorResult nee;
	EstimatorResult bsdf;
	EstimatorResult mis;
};

Comparison compare_all(
	GpuEnvironmentMapHandles const & handles,
	Material const & material,
	f32 const (& wi)[3],
	f32 const rotation,
	u32 const samplesPerThread
) {
	return Comparison {
		.reference = run_reference(handles, material, wi, rotation, 16u),
		.nee = run_estimator(
			handles, material, wi, rotation, 0u, samplesPerThread, 1u
		),
		.bsdf = run_estimator(
			handles, material, wi, rotation, 1u, samplesPerThread, 2u
		),
		.mis = run_estimator(
			handles, material, wi, rotation, 2u, samplesPerThread, 3u
		),
	};
}

// residual quadrature error in the reference itself; the tier-0 case pins
// this at well under 0.5%, so anything inside it is not measurable here
constexpr f64 skToleranceFloor = 0.01;

f64 tolerance_for(EstimatorResult const & measured) {
	return std::fmax(skToleranceFloor, 4.0 * measured.relativeStandardError);
}

void report(char const * const label, Comparison const & c) {
	std::printf(
		"  %-26s ref %.6f | nee %+6.2f%% (+-%.2f) | bsdf %+6.2f%% (+-%.2f)"
		" | mis %+6.2f%% (+-%.2f)\n",
		label,
		luminance(c.reference),
		100.0 * relative_error(c.nee.mean, c.reference),
		100.0 * tolerance_for(c.nee),
		100.0 * relative_error(c.bsdf.mean, c.reference),
		100.0 * tolerance_for(c.bsdf),
		100.0 * relative_error(c.mis.mean, c.reference),
		100.0 * tolerance_for(c.mis)
	);
}

// an estimator agrees with the reference when the gap sits inside its own
// noise floor. a hard sun makes the single-technique estimators genuinely
// noisy, and a fixed percentage cannot tell "under-converged" from "biased"
// -- which is precisely the question this file exists to answer
void check_agreement(
	char const * const label,
	char const * const technique,
	EstimatorResult const & measured,
	Rgb const & reference
) {
	f64 const error = relative_error(measured.mean, reference);
	f64 const tolerance = tolerance_for(measured);
	// doctest stringifies a bare char const * as a pointer
	INFO(
		std::string(label), " / ", std::string(technique),
		": relative error ", error, ", 4-sigma tolerance ", tolerance
	);
	CHECK(measured.nonFiniteCount == 0u);
	CHECK(std::abs(error) < tolerance);
}

void check_all_agree(char const * const label, Comparison const & c) {
	// the MIS estimator is what the renderer actually runs; the two
	// single-technique runs are what localize a failure to one half
	check_agreement(label, "mis", c.mis, c.reference);
	check_agreement(label, "nee", c.nee, c.reference);
	check_agreement(label, "bsdf", c.bsdf, c.reference);
}

// grazing enough to exercise the fresnel/albedo terms, not so grazing that
// the sun falls outside every lobe
constexpr f32 skWi[3] = { 0.6f, 0.8f, 0.0f };

} // namespace

// ---------------------------------------------------------------------------
// -- tier 0: the reference itself
// ---------------------------------------------------------------------------

TEST_CASE("env mis: quadrature reference is converged") {
	auto tables = make_sky(
		ENVIRONMENT_OUTPUT_DIR "env-mis-sun.exr", 40.0f, 4
	);
	REQUIRE(tables.width != 0u);
	auto const handles = handles_from_tables(tables);
	Material const material = make_metal("metal r=0.3", 0.3f);

	// the quadrature integrates the bilinearly reconstructed map; if it is
	// converged, refining the sub-grid must not move it. everything
	// downstream quotes truth against this number, so it is checked first
	Rgb const coarse = run_reference(handles, material, skWi, 0.0f, 8u);
	Rgb const fine = run_reference(handles, material, skWi, 0.0f, 16u);
	Rgb const finer = run_reference(handles, material, skWi, 0.0f, 24u);

	CAPTURE(luminance(coarse));
	CAPTURE(luminance(fine));
	CAPTURE(luminance(finer));
	CHECK(luminance(fine) > 0.0);
	CHECK(std::abs(relative_error(coarse, finer)) < 0.01);
	CHECK(std::abs(relative_error(fine, finer)) < 0.005);

	ponder::environment_tables_destroy(tables);
}

// ---------------------------------------------------------------------------
// -- tier 1: the three estimators must agree, on a map with no bright feature
// ---------------------------------------------------------------------------

TEST_CASE("env mis: smooth map, all three estimators agree") {
	auto tables = make_sky(
		ENVIRONMENT_OUTPUT_DIR "env-mis-smooth.exr", 0.0f, 0
	);
	REQUIRE(tables.width != 0u);
	auto const handles = handles_from_tables(tables);

	// control: with no high-contrast feature envPdf never dominates
	// bsdfPdf, so both halves of the MIS pair carry real weight everywhere
	// and every estimator is well conditioned. a failure here is a plain
	// arithmetic error, not the envPdf-proportional loss the sun cases hunt
	std::printf("env mis -- smooth map (no sun)\n");
	Material const materials[] = {
		make_metal("metal r=0.15", 0.15f),
		make_metal("metal r=0.40", 0.40f),
		make_coated_diffuse("coat r=0.10", 0.10f),
		make_diffuse("diffuse"),
	};
	for (Material const & material : materials) {
		Comparison const c = compare_all(handles, material, skWi, 0.0f, 256u);
		report(material.name, c);
		check_all_agree(material.name, c);
	}

	ponder::environment_tables_destroy(tables);
}

// ---------------------------------------------------------------------------
// -- tier 2: the roughness sweep under a sun -- the headline case
// ---------------------------------------------------------------------------

TEST_CASE("env mis: sun map, roughness sweep, mis must not lose energy") {
	auto tables = make_sky(
		ENVIRONMENT_OUTPUT_DIR "env-mis-sun.exr", 40.0f, 4
	);
	REQUIRE(tables.width != 0u);
	auto const handles = handles_from_tables(tables);

	// roughness is the axis that sets how far envPdf outruns bsdfPdf at the
	// sun: a near-mirror lobe has a huge bsdfPdf and keeps most of its own
	// weight, a broad lobe hands almost everything to the NEE half. if the
	// NEE half under-delivers, the error grows with roughness -- which is
	// exactly the ordering to look for in the printed table
	std::printf("env mis -- sun map, radiance 40, 8x8 texels\n");
	f32 const roughnesses[] = { 0.15f, 0.25f, 0.40f, 0.60f, 1.00f };
	for (f32 const roughness : roughnesses) {
		char label[64];
		std::snprintf(label, sizeof(label), "metal r=%.2f", (f64)roughness);
		Material const material = make_metal(label, roughness);
		Comparison const c = compare_all(handles, material, skWi, 0.0f, 512u);
		report(label, c);
		check_all_agree(label, c);
		// MIS exists to beat the technique that is bad at this map; if it
		// does not, the weights are wrong even when the mean comes out right
		INFO(
			std::string(label), ": mis stddev ", c.mis.luminanceStdDev,
			" vs bsdf ", c.bsdf.luminanceStdDev
		);
		CHECK(c.mis.luminanceStdDev <= c.bsdf.luminanceStdDev);
	}

	ponder::environment_tables_destroy(tables);
}

TEST_CASE("env mis: coat and diffuse under a sun") {
	auto tables = make_sky(
		ENVIRONMENT_OUTPUT_DIR "env-mis-sun.exr", 40.0f, 4
	);
	REQUIRE(tables.width != 0u);
	auto const handles = handles_from_tables(tables);

	// a diffuse lobe has the smallest bsdfPdf of anything here (<= 1/pi),
	// so it leans on the NEE half harder than any metal does -- it should
	// fail first and worst if the NEE half is losing energy
	std::printf("env mis -- sun map, coat + diffuse\n");
	Material const materials[] = {
		make_coated_diffuse("coat r=0.05", 0.05f),
		make_coated_diffuse("coat r=0.20", 0.20f),
		make_diffuse("diffuse"),
	};
	for (Material const & material : materials) {
		Comparison const c = compare_all(handles, material, skWi, 0.0f, 512u);
		report(material.name, c);
		check_all_agree(material.name, c);
	}

	ponder::environment_tables_destroy(tables);
}

TEST_CASE("env mis: rotation invariance") {
	auto tables = make_sky(
		ENVIRONMENT_OUTPUT_DIR "env-mis-sun.exr", 40.0f, 4
	);
	REQUIRE(tables.width != 0u);
	auto const handles = handles_from_tables(tables);

	// rotating the map moves the sun in azimuth. the estimator's error must
	// not depend on where the sun landed: a rotation term dropped from one
	// of ImportanceSample / Pdf / Radiance cancels out of a rotation-0-only
	// test, and the reported symptom was specifically that the dark region
	// tracked the sun as the map was rotated
	std::printf("env mis -- sun map, rotations\n");
	Material const material = make_metal("metal r=0.40", 0.40f);
	f32 const rotations[] = { 0.0f, 1.1f, 2.7f, 4.9f };
	for (f32 const rotation : rotations) {
		char label[64];
		std::snprintf(label, sizeof(label), "rotation %.2f", (f64)rotation);
		Comparison const c = (
			compare_all(handles, material, skWi, rotation, 1024u)
		);
		report(label, c);
		check_all_agree(label, c);
	}

	ponder::environment_tables_destroy(tables);
}

// ---------------------------------------------------------------------------
// -- tier 3: contrast sweep -- how hard a sun the estimator survives
// ---------------------------------------------------------------------------

TEST_CASE("env mis: sun contrast sweep") {
	// a real outdoor HDRI's sun is orders of magnitude above its sky and
	// only a few texels wide. this walks toward that, and prints the error
	// at each step so a threshold failure says *how* the estimator degrades
	// rather than only that it did
	std::printf("env mis -- sun contrast sweep (metal r=0.40)\n");
	Material const material = make_metal("metal r=0.40", 0.40f);
	struct SunCase {
		char const * name;
		f32 radiance;
		i32 halfExtent;
	};
	SunCase const cases[] = {
		{ "sun 40, 8x8", 40.0f, 4 },
		{ "sun 400, 4x4", 400.0f, 2 },
		{ "sun 4000, 2x2", 4000.0f, 1 },
	};
	for (SunCase const & sunCase : cases) {
		auto tables = make_sky(
			ENVIRONMENT_OUTPUT_DIR "env-mis-contrast.exr",
			sunCase.radiance, sunCase.halfExtent
		);
		REQUIRE(tables.width != 0u);
		auto const handles = handles_from_tables(tables);
		Comparison const c = compare_all(handles, material, skWi, 0.0f, 8192u);
		report(sunCase.name, c);
		check_all_agree(sunCase.name, c);
		ponder::environment_tables_destroy(tables);
	}
}

// ---------------------------------------------------------------------------
// -- tier 3b: what a firefly clamp costs each technique
// ---------------------------------------------------------------------------

TEST_CASE("env mis: firefly clamp cost per technique") {
	// resolve.comp caps each written sample's luminance
	// (GpuGlobalExtended.fireflyClampLuminance). That is lossy by
	// construction, but how lossy is diagnostic: env importance sampling
	// exists precisely so that a sun sample arrives with a large pdf and
	// therefore a *modest* estimator value, so a clamp set well above the
	// technique's own mean should cost NEE almost nothing.
	//
	// Loss is reported against each configuration's own unclamped mean, so
	// the columns are comparable even though the nearest-radiance rows are
	// estimating a slightly different integral.
	//
	// The nearest column is the attribution. The pdf table is exact per
	// texel while the production radiance fetch is bilinear, so a texel
	// beside a sun still returns part of that sun while being priced at its
	// own sky density -- a rare, enormous sample. If the clamp loss
	// collapses under a nearest fetch, that mismatch is the mechanism; if it
	// survives, the sub-texel mispricing is
	struct MapCase {
		char const * name;
		f32 radiance;
		i32 halfExtent;
	};
	MapCase const maps[] = {
		{ "smooth (no sun)", 0.0f, 0 },
		{ "sun 40, 8x8", 40.0f, 4 },
		{ "sun 4000, 2x2", 4000.0f, 1 },
	};
	Material const material = make_metal("metal r=0.40", 0.40f);
	f32 const clamps[] = { 100.0f, 10.0f };

	std::printf("env mis -- firefly clamp cost (metal r=0.40)\n");
	for (MapCase const & mapCase : maps) {
		auto tables = make_sky(
			ENVIRONMENT_OUTPUT_DIR "env-mis-clamp.exr",
			mapCase.radiance, mapCase.halfExtent
		);
		REQUIRE(tables.width != 0u);
		auto const handles = handles_from_tables(tables);

		for (bool const nearest : { false, true }) {
			EstimatorResult const neeBase = run_estimator(
				handles, material, skWi, 0.0f, 0u, 2048u, 1u, 0.0f, nearest
			);
			EstimatorResult const misBase = run_estimator(
				handles, material, skWi, 0.0f, 2u, 2048u, 3u, 0.0f, nearest
			);
			for (f32 const clampLuminance : clamps) {
				EstimatorResult const nee = run_estimator(
					handles, material, skWi, 0.0f, 0u, 2048u, 1u,
					clampLuminance, nearest
				);
				EstimatorResult const mis = run_estimator(
					handles, material, skWi, 0.0f, 2u, 2048u, 3u,
					clampLuminance, nearest
				);
				f64 const neeLoss = relative_error(nee.mean, neeBase.mean);
				f64 const misLoss = relative_error(mis.mean, misBase.mean);
				std::printf(
					"  %-16s %-9s clamp %6.1f | nee %+7.2f%% | mis %+7.2f%%\n",
					mapCase.name,
					nearest ? "nearest" : "bilinear",
					(f64)clampLuminance,
					100.0 * neeLoss,
					100.0 * misLoss
				);
				// a clamp set an order of magnitude above the technique's own
				// mean should be nearly free. it is exactly free under a
				// nearest fetch on every map, which is the control that makes
				// the bilinear rows attributable
				INFO(
					std::string(mapCase.name), " ",
					nearest ? "nearest" : "bilinear",
					" clamp ", clampLuminance,
					": nee loss ", neeLoss, ", mis loss ", misLoss
				);
				CHECK(std::abs(neeLoss) < 0.02);
				CHECK(std::abs(misLoss) < 0.02);
			}
		}
		ponder::environment_tables_destroy(tables);
	}
}

// ---------------------------------------------------------------------------
// -- tier 4: sub-texel placement of the drawn sample
// ---------------------------------------------------------------------------

namespace {

struct JitterResult {
	std::vector<u32> binsU;
	std::vector<u32> binsV;
	u32 exactZeroU;
	u32 exactZeroV;
	// samples whose direction does not resolve back to the texel the alias
	// draw picked, so the pdf every consumer recomputes belongs to a
	// different texel than the sample came from
	u32 roundTripMismatch;
};

JitterResult run_jitter(
	GpuEnvironmentMapHandles const & handles,
	u32 const sampleCount,
	u32 const binCount
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR, TEST_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "slang/entry/env_subtexel_jitter.slang",
		.includePaths = srat::slice<char const * const>(includePaths, 2),
	});
	REQUIRE(pl.id != 0);

	u32 const slotCount = binCount * 2u + 3u;
	auto buf = test::make_buffer_u32(slotCount);
	std::vector<u32> const zeros(slotCount, 0u);
	vkof::buffer_upload({
		.buffer = buf,
		.byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(zeros.data()),
			slotCount * sizeof(u32)
		),
	});

	EnvJitterPush const push {
		.binVa = vkof::buffer_virtual_address(buf),
		.handles = handles,
		.sampleCount = sampleCount,
		.binCount = binCount,
		.seedSalt = 7u,
		.rotation = 0.0f,
	};
	static constexpr u32 skJitterLocalSize = 256u;
	test::dispatch(
		pl, push, (sampleCount + skJitterLocalSize - 1u) / skJitterLocalSize
	);

	auto const raw = test::readback<u32>(buf, 0, slotCount);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);

	JitterResult result;
	result.binsU.assign(raw.begin(), raw.begin() + binCount);
	result.binsV.assign(raw.begin() + binCount, raw.begin() + 2 * binCount);
	result.exactZeroU = raw[binCount * 2u + 0u];
	result.exactZeroV = raw[binCount * 2u + 1u];
	result.roundTripMismatch = raw[binCount * 2u + 2u];
	return result;
}

// a flat (uniform) sub-texel distribution has every bin at sampleCount /
// binCount; report the worst bin's deviation from that
f64 max_bin_deviation(std::vector<u32> const & bins, u32 const sampleCount) {
	f64 const expected = (f64)sampleCount / (f64)bins.size();
	f64 worst = 0.0;
	for (u32 const count : bins) {
		worst = std::fmax(worst, std::abs((f64)count - expected) / expected);
	}
	return worst;
}

// builds a smooth map of an arbitrary size; the sub-texel jitter defect is a
// function of width * height alone, so this only needs to vary the texel
// count, not the content
ponder::EnvironmentTables make_sized_sky(
	char const * const path, i32 const width, i32 const height
) {
	std::vector<f32> pixels((size_t)width * height * 4);
	for (i32 y = 0; y < height; ++y) {
		for (i32 x = 0; x < width; ++x) {
			f32 const v = 0.1f + 0.05f * ((f32)x / (f32)width);
			set_pixel(pixels, width, x, y, v, v, v);
		}
	}
	return tables_from_pixels(path, width, height, pixels);
}

} // namespace

TEST_CASE("env mis: sub-texel jitter stays uniform as the map grows") {
	// environmentMapImportanceSample reuses the leftover mantissa bits of
	// xi.x as the continuous position inside the picked texel
	// (util-environment-map.glsl:103-135). f32 has 24 mantissa bits, so
	// once width * height approaches 2^24 the subtraction `nf - float(i)`
	// has nothing left to return and samples pile onto the texel's left
	// edge -- where the bilinear radiance fetch reads the average of two
	// adjacent texels while the pdf still describes the one that was drawn.
	//
	// this asserts the observable directly: the fractional position of the
	// returned direction within its own texel must be uniform in u and v
	std::printf("env mis -- sub-texel jitter uniformity\n");
	constexpr u32 skSampleCount = 1u << 20;
	constexpr u32 skBinCount = 16u;
	struct SizeCase {
		i32 width;
		i32 height;
	};
	SizeCase const cases[] = {
		{ 256, 128 },
		{ 1024, 512 },
		{ 2048, 1024 },
		{ 4096, 2048 },
	};
	for (SizeCase const & sizeCase : cases) {
		auto tables = make_sized_sky(
			ENVIRONMENT_OUTPUT_DIR "env-mis-sized.exr",
			sizeCase.width, sizeCase.height
		);
		REQUIRE(tables.width != 0u);
		auto const handles = handles_from_tables(tables);
		JitterResult const jitter = (
			run_jitter(handles, skSampleCount, skBinCount)
		);
		f64 const devU = max_bin_deviation(jitter.binsU, skSampleCount);
		f64 const devV = max_bin_deviation(jitter.binsV, skSampleCount);
		f64 const zeroFractionU = (f64)jitter.exactZeroU / (f64)skSampleCount;
		f64 const mismatchFraction = (
			(f64)jitter.roundTripMismatch / (f64)skSampleCount
		);
		std::printf(
			"  %5dx%-5d texels %8d | worst u bin %+7.2f%% | worst v bin"
			" %+6.2f%% | fracU == 0 in %5.2f%% | wrong texel %5.3f%%\n",
			sizeCase.width, sizeCase.height,
			sizeCase.width * sizeCase.height,
			100.0 * devU, 100.0 * devV, 100.0 * zeroFractionU,
			100.0 * mismatchFraction
		);
		CAPTURE(sizeCase.width);
		CAPTURE(sizeCase.height);
		CAPTURE(devU);
		CAPTURE(devV);
		CAPTURE(zeroFractionU);
		// a uniform sub-texel position lands 1/binCount of the draws in
		// each bin; 10% slack covers the multinomial noise at 2^20 draws
		CHECK(devU < 0.10);
		CHECK(devV < 0.10);
		// an exact 0.0 is the collapse signature, and is otherwise a
		// measure-zero event
		CHECK(zeroFractionU < 0.01);
		// the pdf a sample is priced with must be the pdf of the texel it
		// was drawn from; anything else breaks the sampler/pdf pairing MIS
		// unbiasedness rests on
		CAPTURE(mismatchFraction);
		CHECK(mismatchFraction < 1e-5);
		ponder::environment_tables_destroy(tables);
	}
}
