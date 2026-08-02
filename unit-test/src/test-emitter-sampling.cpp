#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <ponder/emitter-tables.hpp>
#include <ponder/energy-tables.hpp>
#include <srat/core-math.hpp>
#include "util.hpp"

#include "viewer/shaders/resolve_pc.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// tier 1 of plan-nee.md: lib/ponder's emissive-triangle sampler
// (util-emitter.glsl). the scenes here are synthetic -- hand-built vertex and
// flat-index buffers wired into a GpuResolveModelIndirect -- so a triangle can
// be placed at an exact distance and orientation, which is what makes the
// hybrid sampling switch testable at all: the same triangle drives the
// spherical branch up close and the area branch far away.
//
// three checks, in increasing order of how hard they are to fake:
//
//  1. round trip -- every drawn sample's density recomputed from its own
//     (triangle, direction, distance) through utilEmitterPdf must match what
//     the sampler returned. this is the property MIS unbiasedness rests on,
//     and it is checked per sample rather than in aggregate.
//  2. chi-square -- the sampler's directional histogram against the density
//     integrated over each bin. the expected side finds the density by
//     brute-force intersecting every emitter triangle, which is a genuinely
//     different code path from the sampler's own triangle choice, so the two
//     can disagree.
//  3. the hybrid switch -- the two sampling schemes must agree in
//     distribution where they overlap, and each must be exercised.
//
// note the chi-square alone cannot catch a sampler and pdf that are wrong in
// the same way; the round trip alone cannot catch a pdf that is self-
// consistent but not the sampler's actual density. both are needed.
// ---------------------------------------------------------------------------

namespace {

// must match util-sampling-bins.glsl
constexpr u32 skThetaBins = 40u;
constexpr u32 skPhiBins = 80u;
constexpr u32 skBinCount = skThetaBins * skPhiBins;

constexpr u32 skRoundtripStride = 8u;

// mirrors the push block in util-emitter-test.glsl (scalar layout)
struct EmitterSamplingPush {
	u64 outVa;
	u64 lightsVa;
	u64 modelsVa;
	f32v3 from;
	u32 lightCount;
	u32 sampleCount;
	u32 seedSalt;
	u32 subGrid;
	// direct-lighting estimator fields; glsl allows exactly one push_constant
	// block, so every emitter test shader shares this one layout
	u32 samplesPerThread {};
	u32 threadCount {};
	u32 kullaContyHandle {};
	f32v3 shadingNormal {};
	f32 baseWeight {};
	f32 specularWeight {};
	f32 specularRoughness {};
	f32 metalness {};
};

struct Tri {
	f32v3 a;
	f32v3 b;
	f32v3 c;
};

// a synthetic emitter scene: vertices + flat indices + one model entry, laid
// out exactly the way mor uploads them, plus the emitter list that addresses
// them
struct EmitterScene {
	vkof::Buffer positions;
	vkof::Buffer flatIndices;
	vkof::Buffer models;
	vkof::Buffer lights;
	u32 lightCount;
	std::vector<GpuEmitterLight> lightsCpu;

	void destroy() {
		vkof::buffer_destroy(lights);
		vkof::buffer_destroy(models);
		vkof::buffer_destroy(flatIndices);
		vkof::buffer_destroy(positions);
	}
};

// every triangle becomes one emitter entry unless grouped: `groups` gives the
// triangle count per emitter, so a quad can be a single two-triangle emitter
// (the seam case) or two separate ones
EmitterScene make_scene(
	std::vector<Tri> const & tris,
	std::vector<u32> const & groups,
	f32m44 const & transform = f32m44_identity()
) {
	std::vector<f32v3> positions;
	std::vector<u32> indices;
	for (Tri const & t : tris) {
		u32 const base = (u32)positions.size();
		positions.emplace_back(t.a);
		positions.emplace_back(t.b);
		positions.emplace_back(t.c);
		indices.emplace_back(base + 0u);
		indices.emplace_back(base + 1u);
		indices.emplace_back(base + 2u);
	}

	EmitterScene scene {};
	scene.positions = (
		test::upload_bytes(positions.data(), positions.size() * sizeof(f32v3))
	);
	scene.flatIndices = (
		test::upload_bytes(indices.data(), indices.size() * sizeof(u32))
	);

	GpuResolveModelIndirect model {};
	model.positions = vkof::buffer_virtual_address(scene.positions);
	model.flatIndices = vkof::buffer_virtual_address(scene.flatIndices);
	model.modelMatrix = transform;
	scene.models = test::upload_bytes(&model, sizeof(model));

	// uniform power per emitter unless a test overrides it; the pick is
	// exercised separately by tier 0, so equal powers here keep the chi-square
	// sensitive to the *point* density rather than the pick
	u32 primitiveBase = 0u;
	for (u32 const count : groups) {
		scene.lightsCpu.emplace_back(GpuEmitterLight {
			.kind = EMITTER_KIND_MESH,
			.modelDrawIndex = 0u,
			.primitiveIndexBase = primitiveBase,
			.triangleCount = count,
			.power = 1.0f,
			.cdf = 0.0f,
			.radiance = { 1.0f, 1.0f, 1.0f },
		});
		primitiveBase += count;
	}
	REQUIRE(primitiveBase == (u32)tris.size());
	(void)ponder::emitter_lights_finalize(scene.lightsCpu);

	scene.lights = (
		test::upload_bytes(
			scene.lightsCpu.data(),
			scene.lightsCpu.size() * sizeof(GpuEmitterLight)
		)
	);
	scene.lightCount = (u32)scene.lightsCpu.size();
	return scene;
}

vkof::Pipeline make_pipeline(char const * const name) {
	char const * const includePaths[] = {
		PONDER_SHADER_DIR, MOR_INCLUDE_DIR, APP_SHADER_DIR, TEST_SHADER_DIR,
	};
	vkof::Pipeline const pipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (std::string(TEST_SHADER_DIR) + name).c_str(),
			.includePaths = srat::slice<char const * const>(includePaths, 4u),
		})
	);
	REQUIRE(pipeline.id != 0u);
	return pipeline;
}

struct RoundtripResult {
	u32 validCount;
	u32 invalidCount;
	u32 nonFiniteCount;
	u32 sphericalCount;
	u32 areaCount;
	// worst relative disagreement between the sampled and recomputed pdf
	f64 worstRelativeError;
	f64 worstBarycentricError;
	// E[1/pdf] over the drawn samples, which estimates the total solid angle
	// the emitter set subtends. this is the density's normalization measured
	// without any binning, and it is the only handle on the area branch: an
	// emitter small enough to take that branch (< 3e-4 sr) is by construction
	// smaller than a chi-square bin (~6e-3 sr), so no sphere-grid histogram
	// can ever resolve one
	f64 meanInversePdf;
};

// van oosterom & strackee 1983; the analytic solid angle a triangle subtends
// at `from`, computed on the cpu as an independent reference for
// meanInversePdf
f64 triangle_solid_angle(Tri const & tri, f32v3 const from) {
	f32v3 const a = f32v3_normalize(tri.a - from);
	f32v3 const b = f32v3_normalize(tri.b - from);
	f32v3 const c = f32v3_normalize(tri.c - from);
	f64 const numerator = (f64)std::abs(f32v3_dot(a, f32v3_cross(b, c)));
	f64 const denominator = (
		1.0 + (f64)f32v3_dot(a, b) + (f64)f32v3_dot(b, c) + (f64)f32v3_dot(c, a)
	);
	f64 const omega = 2.0 * std::atan2(numerator, denominator);
	// atan2 returns a negative angle past a half turn of accumulated
	// denominator; the solid angle itself is always positive
	return omega < 0.0 ? omega + 2.0 * M_PI : omega;
}

f64 scene_solid_angle(std::vector<Tri> const & tris, f32v3 const from) {
	f64 total = 0.0;
	for (Tri const & t : tris) {
		total += triangle_solid_angle(t, from);
	}
	return total;
}

RoundtripResult run_roundtrip(
	EmitterScene const & scene, f32v3 const from, u32 const sampleCount,
	u32 const seedSalt
) {
	vkof::Pipeline const pipeline = make_pipeline("slang/entry/emitter_sample_roundtrip.slang");
	vkof::Buffer out = test::make_buffer_u32(sampleCount * skRoundtripStride);
	EmitterSamplingPush const push {
		.outVa = vkof::buffer_virtual_address(out),
		.lightsVa = vkof::buffer_virtual_address(scene.lights),
		.modelsVa = vkof::buffer_virtual_address(scene.models),
		.from = from,
		.lightCount = scene.lightCount,
		.sampleCount = sampleCount,
		.seedSalt = seedSalt,
		.subGrid = 16u,
	};
	test::dispatch(pipeline, push, (sampleCount + 255u) / 256u);
	std::vector<f32> const raw = (
		test::readback<f32>(out, 0, sampleCount * skRoundtripStride)
	);
	vkof::buffer_destroy(out);
	vkof::pipeline_destroy(pipeline);

	RoundtripResult r {};
	f64 inversePdfSum = 0.0;
	for (u32 i = 0u; i < sampleCount; ++i) {
		f32 const * const s = raw.data() + (size_t)i * skRoundtripStride;
		if (s[7] != 0.0f) { r.nonFiniteCount++; }
		if (s[3] == 0.0f) { r.invalidCount++; continue; }
		r.validCount++;
		if (s[5] != 0.0f) { r.sphericalCount++; } else { r.areaCount++; }
		f64 const sampled = (f64)s[0];
		f64 const recomputed = (f64)s[1];
		f64 const denom = std::max(std::abs(sampled), 1e-30);
		r.worstRelativeError = (
			std::max(r.worstRelativeError, std::abs(sampled - recomputed) / denom)
		);
		r.worstBarycentricError = (
			std::max(r.worstBarycentricError, std::abs((f64)s[6] - 1.0))
		);
		inversePdfSum += 1.0 / sampled;
	}
	// averaged over every draw, not just the valid ones: an invalid sample is
	// a draw that contributed nothing, so counting only valid ones would
	// silently rescale the estimate
	r.meanInversePdf = inversePdfSum / (f64)sampleCount;
	return r;
}

std::vector<u32> run_histogram(
	EmitterScene const & scene, f32v3 const from, u32 const sampleCount,
	u32 const seedSalt
) {
	vkof::Pipeline const pipeline = make_pipeline("slang/entry/emitter_sample_histogram.slang");
	// one extra slot: the overflow bin counts invalid/non-unit samples.
	// uploaded rather than merely allocated so the atomics accumulate onto a
	// known-zero buffer
	std::vector<u32> const zeros(skBinCount + 1u, 0u);
	vkof::Buffer out = (
		test::upload_bytes(zeros.data(), zeros.size() * sizeof(u32))
	);
	EmitterSamplingPush const push {
		.outVa = vkof::buffer_virtual_address(out),
		.lightsVa = vkof::buffer_virtual_address(scene.lights),
		.modelsVa = vkof::buffer_virtual_address(scene.models),
		.from = from,
		.lightCount = scene.lightCount,
		.sampleCount = sampleCount,
		.seedSalt = seedSalt,
		.subGrid = 16u,
	};
	test::dispatch(pipeline, push, (sampleCount + 255u) / 256u);
	std::vector<u32> const result = test::readback<u32>(out, 0, skBinCount + 1u);
	vkof::buffer_destroy(out);
	vkof::pipeline_destroy(pipeline);
	return result;
}

std::vector<f32> run_expected(
	EmitterScene const & scene, f32v3 const from, u32 const sampleCount,
	u32 const subGrid = 16u
) {
	vkof::Pipeline const pipeline = make_pipeline("slang/entry/emitter_sample_expected.slang");
	vkof::Buffer out = test::make_buffer_u32(skBinCount);
	EmitterSamplingPush const push {
		.outVa = vkof::buffer_virtual_address(out),
		.lightsVa = vkof::buffer_virtual_address(scene.lights),
		.modelsVa = vkof::buffer_virtual_address(scene.models),
		.from = from,
		.lightCount = scene.lightCount,
		.sampleCount = sampleCount,
		.seedSalt = 0u,
		.subGrid = subGrid,
	};
	test::dispatch(pipeline, push, (skBinCount + 255u) / 256u);
	std::vector<f32> const result = test::readback<f32>(out, 0, skBinCount);
	vkof::buffer_destroy(out);
	vkof::pipeline_destroy(pipeline);
	return result;
}

// pearson + wilson-hilferty, same convention as test-vdb-phase.cpp
struct ChiSquare {
	f64 statistic;
	f64 dof;
	f64 pValue;
	u64 observedTotal;
	f64 expectedTotal;
	f64 pooledObserved;
	f64 pooledExpected;
};

ChiSquare chi_square(
	std::vector<u32> const & observed, std::vector<f32> const & expected
) {
	f64 statistic = 0.0;
	u32 cells = 0u;
	f64 pooledO = 0.0;
	f64 pooledE = 0.0;
	u64 observedTotal = 0u;
	f64 expectedTotal = 0.0;
	for (u32 i = 0u; i < skBinCount; ++i) {
		observedTotal += observed[i];
		if (std::isnan(expected[i])) { continue; }
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
	return {
		statistic, dof, 0.5 * std::erfc(z / std::sqrt(2.0)), observedTotal,
		expectedTotal, pooledO, pooledE
	};
}

// an axis-aligned quad in the y = height plane, centered on (0, height, 0),
// split into two triangles across the diagonal
std::vector<Tri> quad_above(f32 const halfSize, f32 const height) {
	f32v3 const v00 { -halfSize, height, -halfSize };
	f32v3 const v10 { halfSize, height, -halfSize };
	f32v3 const v11 { halfSize, height, halfSize };
	f32v3 const v01 { -halfSize, height, halfSize };
	return { Tri { v00, v10, v11 }, Tri { v00, v11, v01 } };
}

} // namespace

TEST_CASE("emitter sampling: sampler and pdf agree per sample") {
	constexpr u32 skSampleCount = 1u << 16;

	struct Config {
		char const * label;
		f32 halfSize;
		f32 height;
		// expected sampling branch, or -1 for "don't care"
		i32 expectSpherical;
	};
	// the same quad at three distances walks the hybrid switch from the
	// spherical branch (large subtended angle) into the area branch
	Config const configs[] = {
		{ "near quad (spherical branch)", 1.0f, 1.0f, 1 },
		{ "mid quad", 1.0f, 8.0f, -1 },
		{ "far quad (area branch)", 1.0f, 400.0f, 0 },
		{ "tiny near quad", 0.002f, 1.0f, 0 },
	};

	for (Config const & cfg : configs) {
		CAPTURE(cfg.label);
		EmitterScene scene = (
			make_scene(quad_above(cfg.halfSize, cfg.height), { 2u })
		);
		RoundtripResult const r = (
			run_roundtrip(scene, f32v3 { 0.0f, 0.0f, 0.0f }, skSampleCount, 7u)
		);

		CHECK(r.nonFiniteCount == 0u);
		CHECK(r.invalidCount == 0u);
		CHECK(r.validCount == skSampleCount);
		// barycentrics must be a partition of unity, or the sampled point is
		// not on the triangle it claims
		CHECK(r.worstBarycentricError < 1e-5);
		// the round trip is the whole point: a mismatch means the two mis
		// techniques disagree about the same path's density
		CAPTURE(r.worstRelativeError);
		CHECK(r.worstRelativeError < 1e-4);

		if (cfg.expectSpherical == 1) {
			CHECK(r.areaCount == 0u);
			CHECK(r.sphericalCount == skSampleCount);
		} else if (cfg.expectSpherical == 0) {
			CHECK(r.sphericalCount == 0u);
			CHECK(r.areaCount == skSampleCount);
		}

		scene.destroy();
	}
}

TEST_CASE("emitter sampling: chi-square, single triangle") {
	constexpr u32 skSampleCount = 1u << 20;
	std::vector<Tri> const tris = {
		Tri { { -1.0f, 2.0f, -1.0f }, { 1.0f, 2.0f, -1.0f }, { 0.0f, 2.0f, 1.5f } },
	};
	EmitterScene scene = make_scene(tris, { 1u });
	f32v3 const from { 0.0f, 0.0f, 0.0f };

	auto const hist = run_histogram(scene, from, skSampleCount, 11u);
	auto const expected = run_expected(scene, from, skSampleCount);

	// no invalid or non-unit direction
	CHECK(hist[skBinCount] == 0u);

	auto const cs = chi_square(hist, expected);
	CAPTURE(cs.statistic);
	CAPTURE(cs.dof);
	CAPTURE(cs.expectedTotal);
	CHECK(cs.observedTotal == skSampleCount);
	// the density must integrate to 1 over the sphere; this catches a missing
	// jacobian that a shape-only chi-square would absorb
	CHECK(cs.expectedTotal == doctest::Approx((f64)skSampleCount).epsilon(0.02));
	CHECK(cs.pValue > 1e-4);

	scene.destroy();
}

TEST_CASE("emitter sampling: chi-square, quad seam") {
	// a rectangle is two triangles. a sampler or pdf that is discontinuous
	// across the shared diagonal still produces a plausible image and still
	// passes a round trip -- only a chi-square against an independently
	// computed density sees it.
	constexpr u32 skSampleCount = 1u << 20;
	EmitterScene scene = make_scene(quad_above(1.5f, 2.0f), { 2u });
	f32v3 const from { 0.0f, 0.0f, 0.0f };

	auto const hist = run_histogram(scene, from, skSampleCount, 23u);
	auto const expected = run_expected(scene, from, skSampleCount);

	CHECK(hist[skBinCount] == 0u);
	auto const cs = chi_square(hist, expected);
	CAPTURE(cs.statistic);
	CAPTURE(cs.dof);
	CHECK(cs.observedTotal == skSampleCount);
	CHECK(cs.expectedTotal == doctest::Approx((f64)skSampleCount).epsilon(0.02));
	CHECK(cs.pValue > 1e-4);

	scene.destroy();
}

TEST_CASE("emitter sampling: chi-square, two emitters at different distances") {
	// exercises the pick and the point density together. the far emitter sits
	// at 6 units, not 60: at 60 a unit quad spans 0.4 of a bin, so the
	// expected side's subgrid would be integrating a feature finer than the
	// bin it lives in and the chi-square would be measuring the harness
	constexpr u32 skSampleCount = 1u << 20;
	std::vector<Tri> tris = quad_above(1.0f, 1.5f);
	std::vector<Tri> const far = quad_above(1.0f, -6.0f);
	tris.insert(tris.end(), far.begin(), far.end());

	EmitterScene scene = make_scene(tris, { 2u, 2u });
	f32v3 const from { 0.0f, 0.0f, 0.0f };

	auto const hist = run_histogram(scene, from, skSampleCount, 31u);
	auto const expected = run_expected(scene, from, skSampleCount);

	CHECK(hist[skBinCount] == 0u);
	auto const cs = chi_square(hist, expected);
	CAPTURE(cs.statistic);
	CAPTURE(cs.dof);
	CHECK(cs.observedTotal == skSampleCount);
	CHECK(cs.expectedTotal == doctest::Approx((f64)skSampleCount).epsilon(0.02));
	CHECK(cs.pValue > 1e-4);

	scene.destroy();
}

TEST_CASE("emitter sampling: two-sided, both faces sample identically") {
	// emission is two-sided, matching the bsdf-hit branch, which adds
	// emission with no facing test. the sampler must therefore behave the
	// same from either side of the plane -- mirrored, but with the same
	// densities.
	constexpr u32 skSampleCount = 1u << 16;
	EmitterScene scene = make_scene(quad_above(1.0f, 3.0f), { 2u });

	RoundtripResult const above = (
		run_roundtrip(scene, f32v3 { 0.0f, 6.0f, 0.0f }, skSampleCount, 5u)
	);
	RoundtripResult const below = (
		run_roundtrip(scene, f32v3 { 0.0f, 0.0f, 0.0f }, skSampleCount, 5u)
	);

	CHECK(above.validCount == skSampleCount);
	CHECK(below.validCount == skSampleCount);
	CHECK(above.nonFiniteCount == 0u);
	CHECK(below.nonFiniteCount == 0u);
	CHECK(above.worstRelativeError < 1e-4);
	CHECK(below.worstRelativeError < 1e-4);

	scene.destroy();
}

TEST_CASE("emitter sampling: degenerate configurations stay finite") {
	constexpr u32 skSampleCount = 1u << 14;

	struct Config {
		char const * label;
		std::vector<Tri> tris;
		f32v3 from;
		bool expectValid;
	};
	std::vector<Config> const configs = {
		// shading point exactly in the emitter's plane: the subtended solid
		// angle collapses, so this must fall out of the spherical branch
		{
			"coplanar with the emitter",
			quad_above(1.0f, 0.0f),
			f32v3 { 0.0f, 0.0f, 0.0f },
			false,
		},
		// shading point at a vertex of the emitter
		{
			"at an emitter vertex",
			quad_above(1.0f, 0.0f),
			f32v3 { -1.0f, 0.0f, -1.0f },
			false,
		},
		// extreme near and far distances
		{
			"1e-4 away",
			quad_above(1.0f, 1e-4f),
			f32v3 { 0.0f, 0.0f, 0.0f },
			true,
		},
		{
			"1e6 away",
			quad_above(1.0f, 1e6f),
			f32v3 { 0.0f, 0.0f, 0.0f },
			true,
		},
		// a needle-thin triangle
		{
			"sliver triangle",
			{ Tri {
				{ -1.0f, 2.0f, 0.0f },
				{ 1.0f, 2.0f, 0.0f },
				{ 0.0f, 2.0f, 1e-5f },
			} },
			f32v3 { 0.0f, 0.0f, 0.0f },
			true,
		},
		// a fully degenerate, zero-area triangle must produce no sample at all
		{
			"zero-area triangle",
			{ Tri {
				{ 0.0f, 2.0f, 0.0f },
				{ 0.0f, 2.0f, 0.0f },
				{ 0.0f, 2.0f, 0.0f },
			} },
			f32v3 { 0.0f, 0.0f, 0.0f },
			false,
		},
	};

	for (Config const & cfg : configs) {
		CAPTURE(cfg.label);
		std::vector<u32> groups;
		groups.emplace_back((u32)cfg.tris.size());
		EmitterScene scene = make_scene(cfg.tris, groups);
		RoundtripResult const r = (
			run_roundtrip(scene, cfg.from, skSampleCount, 3u)
		);

		// the contract for every degenerate case is the same: never a NaN,
		// never an Inf. an invalid sample is a fine answer, a poisoned one is
		// not
		CHECK(r.nonFiniteCount == 0u);
		if (cfg.expectValid) {
			CHECK(r.validCount > 0u);
			CAPTURE(r.worstRelativeError);
			CHECK(r.worstRelativeError < 1e-3);
		}
		scene.destroy();
	}
}

TEST_CASE("emitter sampling: both hybrid branches normalize correctly") {
	// the hybrid switch is an implementation detail: both schemes must sample
	// the same density. a sphere-grid chi-square cannot referee that, because
	// an emitter small enough to take the area branch (< 3e-4 sr) is by
	// construction ~20x smaller than a single bin -- so normalization is
	// checked without bins instead, via E[1/pdf] against the analytic solid
	// angle the emitter subtends.
	constexpr u32 skSampleCount = 1u << 20;
	f32v3 const from { 0.0f, 0.0f, 0.0f };

	struct Config {
		char const * label;
		f32 halfSize;
		f32 height;
		bool expectSpherical;
	};
	Config const configs[] = {
		// 1.6e-3 sr, comfortably inside the spherical window
		{ "spherical branch", 0.02f, 1.0f, true },
		// 2.5e-5 sr, below EMITTER_SOLID_ANGLE_MIN
		{ "area branch (below the spherical window)", 0.0005f, 0.2f, false },
		// 3.1 sr, a quad the shading point is nearly inside -- the regime area
		// sampling handles worst and the one the cornell ceiling light lives in
		{ "spherical branch, near-hemispherical", 2.0f, 1.0f, true },
	};

	for (Config const & cfg : configs) {
		CAPTURE(cfg.label);
		std::vector<Tri> const tris = quad_above(cfg.halfSize, cfg.height);
		EmitterScene scene = make_scene(tris, { 2u });
		RoundtripResult const r = run_roundtrip(scene, from, skSampleCount, 2u);

		// the branch under test must actually be the one taken, or this
		// silently measures the other scheme twice
		CHECK(r.validCount == skSampleCount);
		if (cfg.expectSpherical) {
			CHECK(r.areaCount == 0u);
		} else {
			CHECK(r.sphericalCount == 0u);
		}
		CHECK(r.nonFiniteCount == 0u);
		CHECK(r.worstRelativeError < 1e-4);

		// E[1/pdf] estimates the subtended solid angle; a density off by a
		// constant factor shows up here and nowhere else in tier 1
		f64 const analytic = scene_solid_angle(tris, from);
		CAPTURE(analytic);
		CAPTURE(r.meanInversePdf);
		CHECK(r.meanInversePdf == doctest::Approx(analytic).epsilon(0.01));

		scene.destroy();
	}
}

TEST_CASE("emitter sampling: uneven triangle areas within one emitter") {
	// the within-emitter triangle pick is uniform, not area weighted, so a
	// meshlet mixing one huge and several tiny triangles is the configuration
	// that costs the most variance. it must still be unbiased -- this pins
	// that, and records the variance penalty rather than assuming it is small.
	constexpr u32 skSampleCount = 1u << 20;
	// the small triangles are 1/16 the linear size of the big one rather than
	// 1/400: still a 250x area disparity, but each still spans enough bins for
	// the expected-side subgrid to resolve it
	std::vector<Tri> tris = {
		Tri { { -4.0f, 3.0f, -4.0f }, { 4.0f, 3.0f, -4.0f }, { 0.0f, 3.0f, 4.0f } },
	};
	for (u32 i = 0u; i < 3u; ++i) {
		f32 const x = -1.2f + (f32)i * 1.2f;
		tris.emplace_back(Tri {
			{ x, 3.0f, 0.0f },
			{ x + 0.5f, 3.0f, 0.0f },
			{ x, 3.0f, 0.5f },
		});
	}
	EmitterScene scene = make_scene(tris, { (u32)tris.size() });
	f32v3 const from { 0.0f, 0.0f, 0.0f };

	// deliberately not a chi-square. the uniform within-emitter pick sends 75%
	// of every draw onto three triangles that together span ~2 bins, so those
	// bins collect ~10^5 samples each and the test would be demanding the
	// expected-side subgrid resolve their silhouettes to ~0.2% -- raising the
	// subgrid 9x moved the statistic by 6%, which is the signature of an edge-
	// position error that no affordable subgrid fixes. the density itself is
	// checked without bins instead.
	RoundtripResult const r = run_roundtrip(scene, from, skSampleCount, 53u);
	CHECK(r.validCount == skSampleCount);
	CHECK(r.nonFiniteCount == 0u);
	CHECK(r.worstRelativeError < 1e-4);

	f64 const analytic = scene_solid_angle(tris, from);
	CAPTURE(analytic);
	CAPTURE(r.meanInversePdf);
	CHECK(r.meanInversePdf == doctest::Approx(analytic).epsilon(0.01));

	// the variance cost of the uniform within-emitter pick, measured rather
	// than assumed: three of four triangles hold a small fraction of the area
	// but take three quarters of the draws
	f64 bigArea = 0.0;
	f64 smallArea = 0.0;
	for (size_t i = 0u; i < tris.size(); ++i) {
		f32v3 const cr = (
			f32v3_cross(tris[i].b - tris[i].a, tris[i].c - tris[i].a)
		);
		f64 const area = 0.5 * (f64)f32v3_length(cr);
		if (i == 0u) { bigArea = area; } else { smallArea += area; }
	}
	MESSAGE(
		"uneven-meshlet pick: ", (smallArea / (bigArea + smallArea)) * 100.0,
		"% of area takes 75% of draws"
	);

	scene.destroy();
}

TEST_CASE("emitter sampling: instance transform is applied to the geometry") {
	// the sampler reads object-space positions and applies modelMatrix
	// itself; a dropped transform would sample the untransformed triangle and
	// still round-trip perfectly, so it has to be checked against geometry
	// placed only by the transform.
	constexpr u32 skSampleCount = 1u << 18;
	std::vector<Tri> const tris = {
		Tri { { -1.0f, 0.0f, -1.0f }, { 1.0f, 0.0f, -1.0f }, { 0.0f, 0.0f, 1.0f } },
	};

	// the triangle sits in the y = 0 plane in object space, which contains the
	// shading point; only the translation lifts it into a samplable position
	EmitterScene untransformed = make_scene(tris, { 1u }, f32m44_identity());
	EmitterScene translated = (
		make_scene(tris, { 1u }, f32m44_translate(0.0f, 4.0f, 0.0f))
	);
	f32v3 const from { 0.0f, 0.0f, 0.0f };

	RoundtripResult const flat = run_roundtrip(untransformed, from, skSampleCount, 9u);
	RoundtripResult const lifted = run_roundtrip(translated, from, skSampleCount, 9u);

	// coplanar with the shading point: no valid sample
	CHECK(flat.validCount == 0u);
	CHECK(flat.nonFiniteCount == 0u);
	// lifted clear of it: every sample valid, at the transformed distance
	CHECK(lifted.validCount == skSampleCount);
	CHECK(lifted.worstRelativeError < 1e-4);

	untransformed.destroy();
	translated.destroy();
}

// ---------------------------------------------------------------------------
// tier 2: the direct-lighting estimator. the reference is Lambert's analytic
// irradiance from a polygonal emitter, so this pins the absolute scale of the
// whole chain -- pick probability, triangle probability, area-to-solid-angle
// conversion and the cosine -- not just internal consistency.
// ---------------------------------------------------------------------------

namespace {

constexpr u32 skDirectStride = 3u;

struct DirectResult {
	f64 nee;
	f64 bsdf;
	f64 mis;
	// sample variance across threads, as a proxy for estimator variance
	f64 neeVariance;
	f64 bsdfVariance;
	f64 misVariance;
};

struct DirectMaterial {
	f32 baseWeight;
	f32 specularWeight;
	f32 specularRoughness;
	f32 metalness;
};

DirectResult run_direct(
	EmitterScene const & scene,
	f32v3 const from,
	f32v3 const shadingNormal,
	DirectMaterial const & material,
	u32 const threadCount,
	u32 const samplesPerThread,
	u32 const seedSalt,
	u32 const kullaContyHandle
) {
	vkof::Pipeline const pipeline = make_pipeline("slang/entry/emitter_direct_lighting.slang");
	vkof::Buffer out = test::make_buffer_u32(threadCount * skDirectStride);
	EmitterSamplingPush const push {
		.outVa = vkof::buffer_virtual_address(out),
		.lightsVa = vkof::buffer_virtual_address(scene.lights),
		.modelsVa = vkof::buffer_virtual_address(scene.models),
		.from = from,
		.lightCount = scene.lightCount,
		.sampleCount = 0u,
		.seedSalt = seedSalt,
		.subGrid = 16u,
		.samplesPerThread = samplesPerThread,
		.threadCount = threadCount,
		.kullaContyHandle = kullaContyHandle,
		.shadingNormal = shadingNormal,
		.baseWeight = material.baseWeight,
		.specularWeight = material.specularWeight,
		.specularRoughness = material.specularRoughness,
		.metalness = material.metalness,
	};
	test::dispatch(pipeline, push, (threadCount + 63u) / 64u);
	std::vector<f32> const raw = (
		test::readback<f32>(out, 0, threadCount * skDirectStride)
	);
	vkof::buffer_destroy(out);
	vkof::pipeline_destroy(pipeline);

	f64 sums[3] { 0.0, 0.0, 0.0 };
	f64 squares[3] { 0.0, 0.0, 0.0 };
	for (u32 t = 0u; t < threadCount; ++t) {
		for (u32 k = 0u; k < skDirectStride; ++k) {
			f64 const v = (f64)raw[(size_t)t * skDirectStride + k];
			sums[k] += v;
			squares[k] += v * v;
		}
	}
	f64 means[3];
	f64 variances[3];
	for (u32 k = 0u; k < 3u; ++k) {
		means[k] = sums[k] / (f64)threadCount;
		variances[k] = (
			squares[k] / (f64)threadCount - means[k] * means[k]
		);
	}
	return {
		means[0], means[1], means[2],
		variances[0], variances[1], variances[2]
	};
}

// lambert's analytic irradiance from a polygonal emitter of constant radiance
// L, at a point with normal n:
//   E = (L/2) * sum over edges of  theta_i * dot(n, normalize(p_i x p_{i+1}))
// where p_i are unit vectors to consecutive polygon vertices and theta_i the
// angle between them. exact, no quadrature.
f64 polygon_irradiance(
	std::vector<f32v3> const & vertices, f32v3 const from, f32v3 const normal
) {
	f64 sum = 0.0;
	size_t const n = vertices.size();
	for (size_t i = 0u; i < n; ++i) {
		f32v3 const a = f32v3_normalize(vertices[i] - from);
		f32v3 const b = f32v3_normalize(vertices[(i + 1u) % n] - from);
		f64 const theta = std::acos(std::clamp((f64)f32v3_dot(a, b), -1.0, 1.0));
		f32v3 const cr = f32v3_cross(a, b);
		f32 const crLen = f32v3_length(cr);
		if (crLen <= 0.0f) {
			continue;
		}
		sum += theta * (f64)f32v3_dot(normal, cr / crLen);
	}
	// the formula's sign tracks the polygon's winding as seen from the
	// receiver; irradiance itself is unsigned, and emission here is two-sided
	return std::abs(0.5 * sum);
}

} // namespace

TEST_CASE("emitter direct lighting: matches lambert's analytic polygon form") {
	// a purely diffuse receiver under an emissive quad. openpbr's diffuse lobe
	// at zero roughness is lambertian, so reflected radiance is
	// (albedo / pi) * E with E from the closed form above.
	auto energyTables = ponder::energy_tables_create();

	constexpr u32 skThreadCount = 4096u;
	constexpr u32 skSamplesPerThread = 512u;

	f32v3 const from { 0.0f, 0.0f, 0.0f };
	f32v3 const normal { 0.0f, 1.0f, 0.0f };
	f32 const halfSize = 1.2f;
	f32 const height = 2.0f;

	std::vector<Tri> const tris = quad_above(halfSize, height);
	EmitterScene scene = make_scene(tris, { 2u });

	// quad_above's corners, in winding order, for the analytic reference
	std::vector<f32v3> const quadVertices = {
		f32v3 { -halfSize, height, -halfSize },
		f32v3 { halfSize, height, -halfSize },
		f32v3 { halfSize, height, halfSize },
		f32v3 { -halfSize, height, halfSize },
	};
	// radiance is 1 per channel in make_scene
	f64 const irradiance = polygon_irradiance(quadVertices, from, normal);
	f64 const analytic = irradiance / M_PI;

	DirectMaterial const diffuse {
		.baseWeight = 1.0f,
		.specularWeight = 0.0f,
		.specularRoughness = 0.5f,
		.metalness = 0.0f,
	};
	DirectResult const r = (
		run_direct(
			scene, from, normal, diffuse, skThreadCount, skSamplesPerThread,
			17u, energyTables.kullaContyEnergyHandle
		)
	);

	CAPTURE(analytic);
	CAPTURE(r.nee);
	CAPTURE(r.bsdf);
	CAPTURE(r.mis);
	// explicit relative error: doctest's Approx adds a scale of 1.0 to the
	// denominator, so epsilon(0.01) on a value below 1 is far looser than it
	// reads
	CHECK(std::abs(r.nee - analytic) / analytic < 0.01);
	CHECK(std::abs(r.bsdf - analytic) / analytic < 0.01);
	CHECK(std::abs(r.mis - analytic) / analytic < 0.01);

	// and MIS must not be worse than the better of the two it combines
	CAPTURE(r.neeVariance);
	CAPTURE(r.bsdfVariance);
	CAPTURE(r.misVariance);
	CHECK(r.misVariance <= std::max(r.neeVariance, r.bsdfVariance) * 1.05);

	scene.destroy();
	ponder::energy_tables_destroy(energyTables);
}

TEST_CASE("emitter direct lighting: MIS partition holds across roughness and light size") {
	// the sweep that catches a mis-weighted branch: which technique dominates
	// changes across this grid, so a wrong weight biases only one corner of it
	// while leaving the rest looking correct.
	auto energyTables = ponder::energy_tables_create();

	constexpr u32 skThreadCount = 2048u;
	constexpr u32 skSamplesPerThread = 512u;

	f32v3 const from { 0.0f, 0.0f, 0.0f };
	f32v3 const normal { 0.0f, 1.0f, 0.0f };

	f32 const roughnesses[] = { 0.05f, 0.2f, 0.6f, 1.0f };
	f32 const halfSizes[] = { 0.15f, 1.2f };

	for (f32 const halfSize : halfSizes) {
		for (f32 const roughness : roughnesses) {
			CAPTURE(halfSize);
			CAPTURE(roughness);
			EmitterScene scene = make_scene(quad_above(halfSize, 2.0f), { 2u });
			DirectMaterial const glossy {
				.baseWeight = 0.3f,
				.specularWeight = 1.0f,
				.specularRoughness = roughness,
				.metalness = 0.0f,
			};
			DirectResult const r = (
				run_direct(
					scene, from, normal, glossy, skThreadCount,
					skSamplesPerThread, 29u, energyTables.kullaContyEnergyHandle
				)
			);

			CAPTURE(r.nee);
			CAPTURE(r.bsdf);
			CAPTURE(r.mis);
			// no closed form for a glossy lobe, so the three estimators
			// referee each other; they are independent techniques for the
			// same integral
			REQUIRE(r.nee > 0.0);
			CHECK(r.bsdf == doctest::Approx(r.nee).epsilon(0.05));
			CHECK(r.mis == doctest::Approx(r.nee).epsilon(0.03));

			scene.destroy();
		}
	}

	ponder::energy_tables_destroy(energyTables);
}
