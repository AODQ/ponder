#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <cmath>
#include <vector>

// tests for openPbrSubsurfaceWalkStep
// (lib/ponder/shaders/util-material-openpbr-subsurface.glsl): one analog
// free-flight step through a bounded homogeneous medium, hero channel
// resampled every step with a probability adaptive to the current
// throughput -- ported 1:1 from blender cycles' random-walk subsurface
// scattering (kernel/closure/volume.h's volume_sample_channel /
// volume_sample_channel_pdf, kernel/integrator/subsurface_random_walk.h's
// per-bounce resample and throughput update).
//
// four self-derived chromatic schemes were tried and rejected before
// reaching for this reference (see the header comment on
// openPbrSubsurfaceWalkStep for the full account): a path-locked hero
// (heavy color noise), weighted delta tracking (multiplicative blowup on
// retries), single-sample ratio tracking against an achromatic mean
// (unbounded variance), and one-sample mis with a *uniformly* chosen
// hero locked per walk (still exponential blowup -- the uniform
// probability meant a fast-colliding hero's own weight sat near its
// bound at nearly every one of its own steps, and (bound * albedo) > 1
// compounds badly over a many-bounce walk). the fix in every case turned
// out to be the same thing cycles already does: make the hero-selection
// probability adaptive (proportional to |throughput * albedo|) instead
// of fixed, so a channel that's already grown large keeps getting
// sampled with the precision it needs, self-stabilizing rather than
// compounding.
//
// unbiasedness doesn't depend on which channel-selection probability is
// used (only variance does -- standard one-sample mis), so the
// unbiasedness tests below hold regardless of the throughput fed in;
// what's new relative to the old uniform-hero test suite is the extra
// `throughput` input and the "no blowup after many compounded steps"
// property, which is really a multi-step property -- covered by
// test-subsurface-furnace.cpp's real-geometry walks, not practical to
// simulate faithfully with this suite's one-thread-per-independent-
// sample dispatch shape.

namespace {

struct WalkParams {
	f32v3 sigmaAbsorption = { 1.0f, 1.0f, 1.0f };
	f32v3 sigmaScattering = { 1.0f, 1.0f, 1.0f };
	f32 distanceMax = 1.0f;
	f32v3 wi = { 0.0f, 0.0f, 1.0f };
	f32 anisotropy = 0.0f;
	f32v3 throughput = { 1.0f, 1.0f, 1.0f };
};

struct WalkPush {
	u64 outcomeVa;
	u64 distanceVa;
	u64 woVa;
	u64 weightVa;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 sigmaAbsorption;
	f32v3 sigmaScattering;
	f32 distanceMax;
	f32v3 wi;
	f32 anisotropy;
	f32v3 throughput;
};
static_assert(sizeof(WalkPush) == 96);

struct WalkResult {
	std::vector<u32> outcome;
	std::vector<f32> distance;
	std::vector<f32v3> wo;
	std::vector<f32v3> weight;
};

constexpr u32 kExit = 0u;
constexpr u32 kScatter = 1u;

WalkResult run_subsurface_walk_step(
	WalkParams const & p, u32 const sampleCount, u32 const seedSalt
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "subsurface_walk_step.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto outcomeBuf = vkof::buffer_create({
		.byteCount = static_cast<u64>(sampleCount) * sizeof(u32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	auto distanceBuf = vkof::buffer_create({
		.byteCount = static_cast<u64>(sampleCount) * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	auto woBuf = vkof::buffer_create({
		.byteCount = static_cast<u64>(sampleCount) * sizeof(f32v3),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	auto weightBuf = vkof::buffer_create({
		.byteCount = static_cast<u64>(sampleCount) * sizeof(f32v3),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	WalkPush const push {
		.outcomeVa = vkof::buffer_virtual_address(outcomeBuf),
		.distanceVa = vkof::buffer_virtual_address(distanceBuf),
		.woVa = vkof::buffer_virtual_address(woBuf),
		.weightVa = vkof::buffer_virtual_address(weightBuf),
		.sampleCount = sampleCount,
		.seedSalt = seedSalt,
		.sigmaAbsorption = p.sigmaAbsorption,
		.sigmaScattering = p.sigmaScattering,
		.distanceMax = p.distanceMax,
		.wi = p.wi,
		.anisotropy = p.anisotropy,
		.throughput = p.throughput,
	};
	static constexpr u32 kLocalSize = 256;
	test::dispatch(pl, push, (sampleCount + kLocalSize - 1u) / kLocalSize);

	WalkResult result;
	result.outcome = test::readback<u32>(outcomeBuf, 0, sampleCount);
	result.distance = test::readback<f32>(distanceBuf, 0, sampleCount);
	result.wo = test::readback<f32v3>(woBuf, 0, sampleCount);
	result.weight = test::readback<f32v3>(weightBuf, 0, sampleCount);

	vkof::buffer_destroy(outcomeBuf);
	vkof::buffer_destroy(distanceBuf);
	vkof::buffer_destroy(woBuf);
	vkof::buffer_destroy(weightBuf);
	vkof::pipeline_destroy(pl);
	return result;
}

} // namespace

TEST_SUITE("[headless]") {

// -----------------------------------------------------------------------------
// task: the whole point of the adaptive weighting -- per-channel weighted
// expectation of "reached distanceMax unscattered, times the accumulated
// correction weight" reproduces that channel's OWN exp(-sigmaT[c]*d),
// regardless of what the (adaptive) channel-selection probability
// actually was. this is standard one-sample mis: unbiasedness doesn't
// depend on which valid technique-probability is used, only variance
// does -- so this holds for cycles' adaptive probability exactly as it
// held for the earlier (rejected for variance, not bias) uniform one.
// -----------------------------------------------------------------------------

TEST_CASE("subsurface walk step: per-channel weighted transmittance is unbiased") {
	constexpr u32 kSampleCount = 1u << 18;
	// deliberately spread far apart, and a non-trivial throughput so the
	// adaptive channel_pdf actually varies per channel instead of falling
	// back to uniform
	WalkParams p;
	p.sigmaAbsorption = { 0.1f, 0.5f, 2.0f };
	p.sigmaScattering = { 0.1f, 0.2f, 0.5f };
	p.distanceMax = 1.0f;
	p.throughput = { 0.8f, 0.3f, 0.05f };
	auto const res = run_subsurface_walk_step(p, kSampleCount, 42u);

	f64 sumX = 0.0, sumY = 0.0, sumZ = 0.0;
	for (u32 i = 0; i < res.outcome.size(); ++i) {
		if (res.outcome[i] != kExit) { continue; }
		sumX += (f64)res.weight[i].x;
		sumY += (f64)res.weight[i].y;
		sumZ += (f64)res.weight[i].z;
	}
	f64 const n = (f64)kSampleCount;
	f64 const meanX = sumX / n;
	f64 const meanY = sumY / n;
	f64 const meanZ = sumZ / n;

	f64 const expectedX = std::exp(
		-((f64)p.sigmaAbsorption.x + (f64)p.sigmaScattering.x) * (f64)p.distanceMax
	);
	f64 const expectedY = std::exp(
		-((f64)p.sigmaAbsorption.y + (f64)p.sigmaScattering.y) * (f64)p.distanceMax
	);
	f64 const expectedZ = std::exp(
		-((f64)p.sigmaAbsorption.z + (f64)p.sigmaScattering.z) * (f64)p.distanceMax
	);

	CAPTURE(meanX); CAPTURE(expectedX);
	CAPTURE(meanY); CAPTURE(expectedY);
	CAPTURE(meanZ); CAPTURE(expectedZ);
	CHECK(std::fabs(meanX - expectedX) < 0.02);
	CHECK(std::fabs(meanY - expectedY) < 0.02);
	CHECK(std::fabs(meanZ - expectedZ) < 0.02);
}

// -----------------------------------------------------------------------------
// task: HG's exact first moment, E[cosTheta] = g, on the sampled re-aim
// direction. the isotropic (g=0) case is checked against E[cosTheta] = 0
// with a moment-based standard error rather than a proportion z-test.
// -----------------------------------------------------------------------------

TEST_CASE("subsurface walk step: mean cosTheta of scatter direction matches g") {
	constexpr u32 kSampleCount = 1u << 18;
	std::vector<f32> const gs { -0.7f, -0.2f, 0.0f, 0.4f, 0.85f };
	for (u32 c = 0; c < gs.size(); ++c) {
		WalkParams p;
		// distanceMax huge: every sample is a real collision
		p.sigmaAbsorption = { 0.0f, 0.0f, 0.0f };
		p.sigmaScattering = { 1.0f, 1.0f, 1.0f };
		p.distanceMax = 1e6f;
		p.wi = { 0.0f, 0.0f, 1.0f };
		p.anisotropy = gs[c];
		auto const res = run_subsurface_walk_step(p, kSampleCount, c + 100u);

		f64 sum = 0.0;
		f64 sumSq = 0.0;
		u64 n = 0;
		for (u32 i = 0; i < res.outcome.size(); ++i) {
			REQUIRE(res.outcome[i] == kScatter);
			f32v3 const wo = res.wo[i];
			f64 const cosTheta = (
				(f64)p.wi.x * wo.x + (f64)p.wi.y * wo.y + (f64)p.wi.z * wo.z
			);
			CHECK(std::isfinite(cosTheta));
			CHECK(std::fabs(std::sqrt(wo.x*wo.x + wo.y*wo.y + wo.z*wo.z) - 1.0f) < 1e-2f);
			sum += cosTheta;
			sumSq += cosTheta * cosTheta;
			n++;
		}
		f64 const mean = sum / (f64)n;
		f64 const variance = std::fmax(sumSq / (f64)n - mean * mean, 0.0);
		f64 const stderrMean = std::sqrt(variance / (f64)n);
		f64 const z = (mean - (f64)gs[c]) / std::fmax(stderrMean, 1e-9);
		CAPTURE(c);
		CAPTURE(gs[c]);
		CAPTURE(mean);
		CAPTURE(z);
		CHECK(std::fabs(z) < 5.0);
	}
}

// -----------------------------------------------------------------------------
// task: degenerate/bounds cases
// -----------------------------------------------------------------------------

TEST_CASE("subsurface walk step: zero extinction always exits at distanceMax") {
	WalkParams p;
	p.sigmaAbsorption = { 0.0f, 0.0f, 0.0f };
	p.sigmaScattering = { 0.0f, 0.0f, 0.0f };
	p.distanceMax = 3.5f;
	auto const res = run_subsurface_walk_step(p, 1024u, 200u);
	for (u32 i = 0; i < res.outcome.size(); ++i) {
		CHECK(res.outcome[i] == kExit);
		CHECK(res.distance[i] == doctest::Approx(3.5f));
		CHECK(res.weight[i].x == doctest::Approx(1.0f));
		CHECK(res.weight[i].y == doctest::Approx(1.0f));
		CHECK(res.weight[i].z == doctest::Approx(1.0f));
	}
}

TEST_CASE("subsurface walk step: zero distanceMax always exits immediately") {
	WalkParams p;
	p.sigmaAbsorption = { 0.5f, 0.5f, 0.5f };
	p.sigmaScattering = { 0.5f, 0.5f, 0.5f };
	p.distanceMax = 0.0f;
	auto const res = run_subsurface_walk_step(p, 1024u, 201u);
	for (u32 i = 0; i < res.outcome.size(); ++i) {
		CHECK(res.outcome[i] == kExit);
		CHECK(res.distance[i] == doctest::Approx(0.0f));
	}
}

TEST_CASE("subsurface walk step: zero throughput falls back to uniform channel pdf, stays finite") {
	// openPbrSubsurfaceChannelPdf's fallback branch: throughput=0 means
	// |throughput*albedo| sums to 0 for every channel, which must not
	// divide by zero
	WalkParams p;
	p.sigmaAbsorption = { 0.3f, 0.6f, 0.9f };
	p.sigmaScattering = { 0.3f, 0.2f, 0.1f };
	p.distanceMax = 1.0f;
	p.throughput = { 0.0f, 0.0f, 0.0f };
	auto const res = run_subsurface_walk_step(p, 4096u, 250u);
	u32 nonFinite = 0;
	for (u32 i = 0; i < res.outcome.size(); ++i) {
		f32v3 const w = res.weight[i];
		if (!std::isfinite(w.x) || !std::isfinite(w.y) || !std::isfinite(w.z)) {
			nonFinite++;
		}
	}
	CHECK(nonFinite == 0u);
}

TEST_CASE("subsurface walk step: extreme anisotropy stays finite and unit-length") {
	std::vector<f32> const gs { -0.999f, -0.95f, 0.95f, 0.999f };
	for (u32 c = 0; c < gs.size(); ++c) {
		WalkParams p;
		p.sigmaAbsorption = { 0.0f, 0.0f, 0.0f };
		p.sigmaScattering = { 2.0f, 2.0f, 2.0f };
		p.distanceMax = 1e6f;
		p.anisotropy = gs[c];
		auto const res = run_subsurface_walk_step(p, 4096u, c + 300u);
		u32 nonFinite = 0;
		u32 nonUnit = 0;
		for (u32 i = 0; i < res.outcome.size(); ++i) {
			f32v3 const wo = res.wo[i];
			bool const finite = (
				std::isfinite(wo.x) && std::isfinite(wo.y) && std::isfinite(wo.z)
			);
			if (!finite) { nonFinite++; continue; }
			f32 const len = std::sqrt(wo.x*wo.x + wo.y*wo.y + wo.z*wo.z);
			if (std::fabs(len - 1.0f) > 1e-2f) { nonUnit++; }
		}
		CAPTURE(gs[c]);
		CHECK(nonFinite == 0u);
		CHECK(nonUnit == 0u);
	}
}

TEST_CASE("subsurface walk step: extreme chromatic contrast keeps weight finite") {
	// same extreme spread that produced nan pixels with the earlier
	// (rejected) uniform-per-walk-hero scheme
	WalkParams p;
	p.sigmaAbsorption = { 1e-4f, 0.5f, 50.0f };
	p.sigmaScattering = { 1e-4f, 0.5f, 50.0f };
	p.distanceMax = 2.0f;
	p.throughput = { 1.0f, 1.0f, 1.0f };
	auto const res = run_subsurface_walk_step(p, 8192u, 400u);
	u32 nonFinite = 0;
	for (u32 i = 0; i < res.outcome.size(); ++i) {
		f32v3 const w = res.weight[i];
		if (!std::isfinite(w.x) || !std::isfinite(w.y) || !std::isfinite(w.z)) {
			nonFinite++;
		}
	}
	CHECK(nonFinite == 0u);
}

} // TEST_SUITE("[headless]")
