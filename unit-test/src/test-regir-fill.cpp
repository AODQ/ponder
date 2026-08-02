#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <ponder/emitter-tables.hpp>
#include <srat/core-math.hpp>
#include "util.hpp"

#include "viewer/shaders/resolve_pc.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// tier 10 of plan-many-lights.md: resampled importance sampling and the
// reservoir, tested without any grid.
//
// utilRegirFillReservoir takes a representative point and normal rather than a
// cell, so none of this needs the hash table, the scout pass or a clear pass to
// exist. tier 9 owns the table; this file owns the math; neither depends on the
// other.
//
// **the one property everything here rests on.** ris returns an unbiased
// contribution weight W in place of a pdf, defined so that
//
//     E[W | Y = x] = 1 / p_true(x)
//
// where p_true is the (unknown, not closed form) density ris actually induces.
// You cannot check that directly -- p_true has no expression. What you can
// check is its consequence:
//
//     E[f(Y) * W_Y] = integral over the light surface of f(x) dA(x)
//
// for *any* f. Two choices of f make that concrete and are what the cases below
// use:
//
//   f == 1        -> E[W] is the emitter's total world-space surface area.
//                    exactly computable from three vertices. this is the
//                    headline test: it fails for a missing 1/M, a wrong
//                    acceptance probability, a measure mix-up, or a target
//                    function inconsistent between the weight and the survivor.
//   f == \hat{h}  -> E[targetValue * W] is \int \hat{h} dA, which the fill loop
//                    *also* estimates as its normalization factor by summing
//                    \hat{h}/p directly. two independent routes to one number.
//
// a third, blunter check: \hat{h} is arbitrary, so scaling it by a constant
// must not move any of the above. that catches a normalization baked into the
// wrong place.
// ---------------------------------------------------------------------------

namespace {

// must match regir_fill_probe.comp
constexpr u32 skProbeStride = 8u;
// must match util-regir-shared.h
constexpr u32 skCandidateCount = 32u;

// mirrors the push block in regir_fill_probe.comp (scalar layout)
struct FillProbePush {
	u64 outVa;
	u64 lightsVa;
	u64 modelsVa;
	f32v3 representativePosition;
	f32v3 representativeNormal;
	u32 lightCount;
	u32 realizationCount;
	u32 seedSalt;
	u32 hasNormal;
};

struct Tri {
	f32v3 a;
	f32v3 b;
	f32v3 c;
};

f32v3 sub(f32v3 const a, f32v3 const b) {
	return f32v3 { a.x - b.x, a.y - b.y, a.z - b.z };
}

f32v3 cross3(f32v3 const a, f32v3 const b) {
	return f32v3 {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x,
	};
}

f64 dot3(f32v3 const a, f32v3 const b) {
	return (f64)a.x * (f64)b.x + (f64)a.y * (f64)b.y + (f64)a.z * (f64)b.z;
}

f64 length3(f32v3 const v) {
	return std::sqrt(dot3(v, v));
}

f64 triangle_area(Tri const & t) {
	return 0.5 * length3(cross3(sub(t.b, t.a), sub(t.c, t.a)));
}

f32v3 triangle_normal(Tri const & t) {
	f32v3 const c = cross3(sub(t.b, t.a), sub(t.c, t.a));
	f64 const len = length3(c);
	if (len <= 0.0) {
		return f32v3 { 0.0f, 0.0f, 0.0f };
	}
	return f32v3 { (f32)(c.x / len), (f32)(c.y / len), (f32)(c.z / len) };
}

// the same target function regir_fill_probe.comp evaluates, in f64, integrated
// over a triangle by uniform subdivision. this is the independent reference for
// \int \hat{h} dA -- deliberately a different method from anything the shader
// does, so a shared mistake cannot hide
f64 target_integral(
	Tri const & t,
	f32v3 const & from,
	f32v3 const & normal,
	bool const hasNormal,
	f64 const emitted,
	u32 const subdivisions
) {
	f32v3 const triNormal = triangle_normal(t);
	f64 const area = triangle_area(t);
	f64 const cellArea = area / (f64)(subdivisions * subdivisions);
	f64 total = 0.0;
	// barycentric sub-triangle centroids on a regular subdivision; every
	// sub-triangle has identical area, so a plain sum times cellArea is the
	// midpoint rule
	for (u32 i = 0u; i < subdivisions; ++i) {
		for (u32 j = 0u; j + i < subdivisions; ++j) {
			// two orientations per lattice cell, except the last column
			u32 const orientations = (j + i + 1u < subdivisions) ? 2u : 1u;
			for (u32 o = 0u; o < orientations; ++o) {
				f64 const inv = 1.0 / (f64)subdivisions;
				f64 b1;
				f64 b2;
				if (o == 0u) {
					b1 = ((f64)i + 1.0 / 3.0) * inv;
					b2 = ((f64)j + 1.0 / 3.0) * inv;
				} else {
					b1 = ((f64)i + 2.0 / 3.0) * inv;
					b2 = ((f64)j + 2.0 / 3.0) * inv;
				}
				f64 const b0 = 1.0 - b1 - b2;
				f32v3 const p {
					(f32)(b0 * t.a.x + b1 * t.b.x + b2 * t.c.x),
					(f32)(b0 * t.a.y + b1 * t.b.y + b2 * t.c.y),
					(f32)(b0 * t.a.z + b1 * t.b.z + b2 * t.c.z),
				};
				f32v3 const toLight = sub(p, from);
				f64 const distanceSq = dot3(toLight, toLight);
				if (distanceSq <= 0.0) {
					continue;
				}
				f64 const distance = std::sqrt(distanceSq);
				f32v3 const dir {
					(f32)(toLight.x / distance),
					(f32)(toLight.y / distance),
					(f32)(toLight.z / distance),
				};
				f64 const cosAtLight = std::abs(dot3(triNormal, dir));
				f64 const cosAtSurface = (
					hasNormal ? std::max(dot3(normal, dir), 0.0) : 1.0
				);
				total += (
					emitted * cosAtLight * cosAtSurface / distanceSq * cellArea
				);
			}
		}
	}
	return total;
}

// a synthetic emitter scene, same shape as test-emitter-sampling.cpp's: raw
// vertex/index buffers plus a one-entry model, so a triangle can be placed at
// an exact distance and orientation
struct Scene {
	vkof::Buffer positions {};
	vkof::Buffer flatIndices {};
	vkof::Buffer models {};
	vkof::Buffer lights {};
	u32 lightCount {};
	std::vector<GpuEmitterLight> lightsCpu;

	void destroy() {
		if (lights.id != 0u) { vkof::buffer_destroy(lights); }
		if (models.id != 0u) { vkof::buffer_destroy(models); }
		if (flatIndices.id != 0u) { vkof::buffer_destroy(flatIndices); }
		if (positions.id != 0u) { vkof::buffer_destroy(positions); }
	}
};

Scene make_scene(
	std::vector<Tri> const & tris,
	std::vector<u32> const & groups,
	f32v3 const & radiance = f32v3 { 1.0f, 1.0f, 1.0f }
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

	Scene scene {};
	scene.positions = (
		test::upload_bytes(positions.data(), positions.size() * sizeof(f32v3))
	);
	scene.flatIndices = (
		test::upload_bytes(indices.data(), indices.size() * sizeof(u32))
	);

	GpuResolveModelIndirect model {};
	model.positions = vkof::buffer_virtual_address(scene.positions);
	model.flatIndices = vkof::buffer_virtual_address(scene.flatIndices);
	model.modelMatrix = f32m44_identity();
	scene.models = test::upload_bytes(&model, sizeof(model));

	u32 primitiveBase = 0u;
	for (u32 const count : groups) {
		scene.lightsCpu.emplace_back(GpuEmitterLight {
			.kind = EMITTER_KIND_MESH,
			.modelDrawIndex = 0u,
			.primitiveIndexBase = primitiveBase,
			.triangleCount = count,
			.power = 1.0f,
			.cdf = 0.0f,
			.radiance = radiance,
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

vkof::Pipeline make_pipeline() {
	char const * const includePaths[] = {
		PONDER_SHADER_DIR, MOR_INCLUDE_DIR, APP_SHADER_DIR, TEST_SHADER_DIR,
	};
	vkof::Pipeline const pipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				std::string(TEST_SHADER_DIR) + "slang/entry/regir_fill_probe.slang"
			).c_str(),
			.includePaths = srat::slice<char const * const>(includePaths, 4u),
		})
	);
	REQUIRE(pipeline.id != 0u);
	return pipeline;
}

struct ScopedPipeline {
	vkof::Pipeline handle {};

	ScopedPipeline() : handle(make_pipeline()) {}
	~ScopedPipeline() {
		if (handle.id != 0u) {
			vkof::pipeline_destroy(handle);
		}
	}
	ScopedPipeline(ScopedPipeline const &) = delete;
	ScopedPipeline & operator=(ScopedPipeline const &) = delete;

	operator vkof::Pipeline() const { return handle; }
};

struct ScopedScene {
	Scene scene;

	explicit ScopedScene(Scene s) : scene(std::move(s)) {}
	~ScopedScene() { scene.destroy(); }
	ScopedScene(ScopedScene const &) = delete;
	ScopedScene & operator=(ScopedScene const &) = delete;
};

// aggregate statistics over many independent cell-fill realizations
struct FillStats {
	u32 realizations;
	u32 validCount;
	u32 nonFiniteCount;
	// mean of W over *all* realizations, including the ones whose survivor was
	// invalid (W == 0). those must stay in the average -- dropping them is
	// conditioning on the outcome and biases the estimate upward
	f64 meanWeight;
	f64 meanTargetTimesWeight;
	// mean of (weightSum / M): the fill pass's own normalization estimate
	f64 meanNormalization;
	f64 stdErrWeight;
};

FillStats run_fill(
	vkof::Pipeline const pipeline,
	Scene const & scene,
	f32v3 const & from,
	f32v3 const & normal,
	bool const hasNormal,
	u32 const realizations,
	u32 const seedSalt
) {
	vkof::Buffer const outBuf = (
		test::make_buffer_u32(realizations * skProbeStride)
	);
	FillProbePush const push {
		.outVa = vkof::buffer_virtual_address(outBuf),
		.lightsVa = vkof::buffer_virtual_address(scene.lights),
		.modelsVa = vkof::buffer_virtual_address(scene.models),
		.representativePosition = from,
		.representativeNormal = normal,
		.lightCount = scene.lightCount,
		.realizationCount = realizations,
		.seedSalt = seedSalt,
		.hasNormal = hasNormal ? 1u : 0u,
	};
	test::dispatch(pipeline, push, (realizations + 63u) / 64u);

	std::vector<f32> const raw = (
		test::readback<f32>(outBuf, 0u, realizations * skProbeStride)
	);
	vkof::buffer_destroy(outBuf);

	FillStats stats {};
	stats.realizations = realizations;
	f64 weightSum = 0.0;
	f64 weightSqSum = 0.0;
	f64 targetWeightSum = 0.0;
	f64 normSum = 0.0;
	for (u32 i = 0u; i < realizations; ++i) {
		f32 const w = raw[i * skProbeStride + 0u];
		f32 const target = raw[i * skProbeStride + 1u];
		f32 const rawWeightSum = raw[i * skProbeStride + 2u];
		f32 const valid = raw[i * skProbeStride + 7u];
		if (!std::isfinite(w) || !std::isfinite(target)
			|| !std::isfinite(rawWeightSum)) {
			stats.nonFiniteCount += 1u;
			continue;
		}
		if (valid > 0.5f) {
			stats.validCount += 1u;
		}
		weightSum += (f64)w;
		weightSqSum += (f64)w * (f64)w;
		targetWeightSum += (f64)target * (f64)w;
		normSum += (f64)rawWeightSum / (f64)skCandidateCount;
	}
	f64 const n = (f64)realizations;
	stats.meanWeight = weightSum / n;
	stats.meanTargetTimesWeight = targetWeightSum / n;
	stats.meanNormalization = normSum / n;
	f64 const variance = (
		std::max(weightSqSum / n - stats.meanWeight * stats.meanWeight, 0.0)
	);
	stats.stdErrWeight = std::sqrt(variance / n);
	return stats;
}


// -- 10.6 helpers: fill a cell, then shade from it at a different point -------

// must match util-regir-shared.h
constexpr u32 skReservoirCount = 64u;

// mirrors the push block in regir_direct_lighting.comp (scalar layout)
struct DirectProbePush {
	u64 outVa;
	u64 reservoirsVa;
	u64 lightsVa;
	u64 modelsVa;
	f32v3 representativePosition;
	f32v3 representativeNormal;
	f32v3 shadingPosition;
	f32v3 shadingNormal;
	u32 lightCount;
	u32 sampleCount;
	u32 seedSalt;
	u32 mode;
};

constexpr u32 skDirectModeFill = 0u;
constexpr u32 skDirectModeShade = 1u;

vkof::Pipeline make_direct_pipeline() {
	char const * const includePaths[] = {
		PONDER_SHADER_DIR, MOR_INCLUDE_DIR, APP_SHADER_DIR, TEST_SHADER_DIR,
	};
	vkof::Pipeline const pipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				std::string(TEST_SHADER_DIR) + "slang/entry/regir_direct_lighting.slang"
			).c_str(),
			.includePaths = srat::slice<char const * const>(includePaths, 4u),
		})
	);
	REQUIRE(pipeline.id != 0u);
	return pipeline;
}

struct ScopedDirectPipeline {
	vkof::Pipeline handle {};

	ScopedDirectPipeline() : handle(make_direct_pipeline()) {}
	~ScopedDirectPipeline() {
		if (handle.id != 0u) {
			vkof::pipeline_destroy(handle);
		}
	}
	ScopedDirectPipeline(ScopedDirectPipeline const &) = delete;
	ScopedDirectPipeline & operator=(ScopedDirectPipeline const &) = delete;

	operator vkof::Pipeline() const { return handle; }
};

struct DirectStats {
	u32 nonFiniteCount;
	f64 mean;
	f64 stdErr;
};

// two dispatches with the reservoir array between them, exactly the ordering
// the renderer uses: fill runs once for the cell, shade runs per sample and
// only reads
DirectStats run_direct(
	vkof::Pipeline const pipeline,
	Scene const & scene,
	f32v3 const & representativePosition,
	f32v3 const & representativeNormal,
	f32v3 const & shadingPosition,
	f32v3 const & shadingNormal,
	u32 const sampleCount,
	u32 const fillRealizations,
	u32 const seedSalt
) {
	vkof::Buffer const reservoirs = (
		vkof::buffer_create({
			.byteCount = (
				(u64)skReservoirCount * sizeof(GpuRegirReservoir)
			),
			.memory = vkof::BufferMemory::DeviceOnly,
		})
	);
	vkof::Buffer const outBuf = test::make_buffer_u32(sampleCount);

	DirectStats stats {};
	// one realization is a *whole fill* plus the shading draws that read it.
	// the mean and standard error are taken across realizations, never across
	// the draws inside one: W's unbiasedness is an expectation over fill
	// randomness, so shading harder against a frozen fill converges to that
	// fill's own answer rather than to the integral. averaging draws and
	// calling their spread the error bar reports a precision that does not
	// exist -- measured, at offset 0.5, as an apparent 85 sigma discrepancy
	// that was entirely fill noise
	f64 realizationSum = 0.0;
	f64 realizationSqSum = 0.0;
	for (u32 r = 0u; r < fillRealizations; ++r) {
		DirectProbePush push {
			.outVa = vkof::buffer_virtual_address(outBuf),
			.reservoirsVa = vkof::buffer_virtual_address(reservoirs),
			.lightsVa = vkof::buffer_virtual_address(scene.lights),
			.modelsVa = vkof::buffer_virtual_address(scene.models),
			.representativePosition = representativePosition,
			.representativeNormal = representativeNormal,
			.shadingPosition = shadingPosition,
			.shadingNormal = shadingNormal,
			.lightCount = scene.lightCount,
			.sampleCount = sampleCount,
			.seedSalt = seedSalt + r * 0x9E3779B9u,
			.mode = skDirectModeFill,
		};
		test::dispatch(pipeline, push, 1u);

		push.mode = skDirectModeShade;
		test::dispatch(pipeline, push, (sampleCount + 63u) / 64u);

		std::vector<f32> const raw = (
			test::readback<f32>(outBuf, 0u, sampleCount)
		);

		f64 sum = 0.0;
		for (u32 i = 0u; i < sampleCount; ++i) {
			f32 const v = raw[i];
			if (!std::isfinite(v)) {
				stats.nonFiniteCount += 1u;
				continue;
			}
			sum += (f64)v;
		}
		f64 const realizationMean = sum / (f64)sampleCount;
		realizationSum += realizationMean;
		realizationSqSum += realizationMean * realizationMean;
	}
	vkof::buffer_destroy(outBuf);
	vkof::buffer_destroy(reservoirs);

	f64 const n = (f64)fillRealizations;
	stats.mean = realizationSum / n;
	f64 const variance = (
		std::max(realizationSqSum / n - stats.mean * stats.mean, 0.0)
	);
	stats.stdErr = std::sqrt(variance / n);
	return stats;
}

} // namespace

// ---------------------------------------------------------------------------
// 10.1 -- the headline: E[W] is the emitter's surface area
//
// with f == 1 the unbiasedness identity reads E[W] = \int 1 dA = total area.
// nothing about the target function, the source pdf, the candidate count or
// the acceptance test appears in that answer, which is exactly what makes it a
// good gate: every one of those being wrong moves E[W] and none of them can
// move it *back*.
// ---------------------------------------------------------------------------

TEST_CASE("regir fill: E[W] equals the emitter area") {
	ScopedPipeline const pipeline;

	struct Config {
		char const * name;
		Tri tri;
		f32v3 from;
		f32v3 normal;
	};

	// spanning both sampling branches of the underlying emitter sampler: a
	// close triangle takes the spherical-triangle path, a distant one falls
	// back to uniform area sampling. W has to come out the same either way
	Config const configs[] = {
		{
			"near, facing",
			Tri {
				f32v3 { -0.5f, 1.0f, -0.5f },
				f32v3 { 0.5f, 1.0f, -0.5f },
				f32v3 { -0.5f, 1.0f, 0.5f },
			},
			f32v3 { 0.0f, 0.0f, 0.0f },
			f32v3 { 0.0f, 1.0f, 0.0f },
		},
		{
			"far, facing",
			Tri {
				f32v3 { -0.5f, 40.0f, -0.5f },
				f32v3 { 0.5f, 40.0f, -0.5f },
				f32v3 { -0.5f, 40.0f, 0.5f },
			},
			f32v3 { 0.0f, 0.0f, 0.0f },
			f32v3 { 0.0f, 1.0f, 0.0f },
		},
		{
			"large, close",
			Tri {
				f32v3 { -8.0f, 2.0f, -8.0f },
				f32v3 { 8.0f, 2.0f, -8.0f },
				f32v3 { -8.0f, 2.0f, 8.0f },
			},
			f32v3 { 0.0f, 0.0f, 0.0f },
			f32v3 { 0.0f, 1.0f, 0.0f },
		},
		{
			"oblique receiver",
			Tri {
				f32v3 { -0.5f, 1.0f, -0.5f },
				f32v3 { 0.5f, 1.0f, -0.5f },
				f32v3 { -0.5f, 1.0f, 0.5f },
			},
			f32v3 { 0.0f, 0.0f, 0.0f },
			f32v3 { 0.6f, 0.8f, 0.0f },
		},
	};

	constexpr u32 skRealizations = 200000u;

	for (Config const & c : configs) {
		CAPTURE(c.name);
		ScopedScene const scene(make_scene({ c.tri }, { 1u }));
		FillStats const stats = (
			run_fill(
				pipeline, scene.scene, c.from, c.normal, /*hasNormal=*/true,
				skRealizations, 0x51ED0001u
			)
		);

		CHECK(stats.nonFiniteCount == 0u);
		REQUIRE(stats.validCount > 0u);

		f64 const area = triangle_area(c.tri);
		f64 const relative = (stats.meanWeight - area) / area;
		f64 const relativeSigma = stats.stdErrWeight / area;
		CAPTURE(area);
		CAPTURE(stats.meanWeight);
		CAPTURE(relative);
		CAPTURE(relativeSigma);
		// gate on the measured standard error rather than a fixed tolerance:
		// W's variance depends heavily on how well the target matches the
		// source, which differs a lot across these configurations
		CHECK(std::abs(relative) < std::max(5.0 * relativeSigma, 0.01));
	}
}

TEST_CASE("regir fill: E[W] is the summed area over several emitters") {
	ScopedPipeline const pipeline;

	// three triangles at different distances and orientations, as three
	// separate emitter entries, so the emitter pick is genuinely exercised
	// rather than being a foregone conclusion
	std::vector<Tri> const tris {
		Tri {
			f32v3 { -0.5f, 1.0f, -0.5f },
			f32v3 { 0.5f, 1.0f, -0.5f },
			f32v3 { -0.5f, 1.0f, 0.5f },
		},
		Tri {
			f32v3 { 2.0f, 3.0f, -1.0f },
			f32v3 { 3.0f, 3.0f, -1.0f },
			f32v3 { 2.0f, 3.0f, 1.0f },
		},
		Tri {
			f32v3 { -6.0f, 12.0f, -6.0f },
			f32v3 { 6.0f, 12.0f, -6.0f },
			f32v3 { -6.0f, 12.0f, 6.0f },
		},
	};
	ScopedScene const scene(make_scene(tris, { 1u, 1u, 1u }));

	f64 totalArea = 0.0;
	for (Tri const & t : tris) {
		totalArea += triangle_area(t);
	}

	FillStats const stats = (
		run_fill(
			pipeline, scene.scene,
			f32v3 { 0.0f, 0.0f, 0.0f }, f32v3 { 0.0f, 1.0f, 0.0f },
			/*hasNormal=*/true, 400000u, 0x51ED0002u
		)
	);

	CHECK(stats.nonFiniteCount == 0u);
	f64 const relative = (stats.meanWeight - totalArea) / totalArea;
	f64 const relativeSigma = stats.stdErrWeight / totalArea;
	CAPTURE(totalArea);
	CAPTURE(stats.meanWeight);
	CAPTURE(relative);
	CAPTURE(relativeSigma);
	CHECK(std::abs(relative) < std::max(5.0 * relativeSigma, 0.01));
}

// ---------------------------------------------------------------------------
// 10.2 -- the normalization factor
//
// two independent estimates of \int \hat{h} dA must agree with each other and
// with cpu quadrature:
//
//   a) the fill pass's own sum of \hat{h}/p over its candidates (this is what
//      gets stored on the cell and later turns \hat{h} into a pdf)
//   b) E[\hat{h}(Y) * W_Y], the unbiasedness identity with f == \hat{h}
//   c) midpoint-rule quadrature over the triangle, in f64
//
// (a) and (b) sharing a bug is plausible -- they come from the same loop. (c)
// shares nothing with either.
// ---------------------------------------------------------------------------

TEST_CASE("regir fill: the normalization factor is the target integral") {
	ScopedPipeline const pipeline;

	Tri const tri {
		f32v3 { -0.5f, 1.0f, -0.5f },
		f32v3 { 0.5f, 1.0f, -0.5f },
		f32v3 { -0.5f, 1.0f, 0.5f },
	};
	f32v3 const from { 0.0f, 0.0f, 0.0f };
	f32v3 const normal { 0.0f, 1.0f, 0.0f };

	ScopedScene const scene(make_scene({ tri }, { 1u }));
	FillStats const stats = (
		run_fill(
			pipeline, scene.scene, from, normal, /*hasNormal=*/true,
			400000u, 0x51ED0003u
		)
	);

	// radiance is (1,1,1) and \hat{h} uses rec709 luminance, which sums to 1
	f64 const reference = (
		target_integral(tri, from, normal, /*hasNormal=*/true, 1.0, 96u)
	);
	CAPTURE(reference);
	CAPTURE(stats.meanNormalization);
	CAPTURE(stats.meanTargetTimesWeight);

	CHECK(stats.nonFiniteCount == 0u);
	CHECK(
		std::abs(stats.meanNormalization - reference) / reference < 0.02
	);
	CHECK(
		std::abs(stats.meanTargetTimesWeight - reference) / reference < 0.02
	);
}

// ---------------------------------------------------------------------------
// 10.3 -- the target function is arbitrary
//
// nothing about unbiasedness depends on \hat{h} being any particular function,
// only on its support covering the integrand's. so scaling \hat{h} by a
// constant -- here by scaling the emitter's radiance, which multiplies \hat{h}
// and nothing else -- must leave E[W] exactly where it was, while moving the
// normalization factor in exact proportion.
//
// this is the test that catches a normalization folded into the wrong place:
// a stray target value in W, or a missing one, shows up as E[W] tracking the
// scale instead of ignoring it.
// ---------------------------------------------------------------------------

TEST_CASE("regir fill: scaling the target leaves E[W] alone") {
	ScopedPipeline const pipeline;

	Tri const tri {
		f32v3 { -0.5f, 1.0f, -0.5f },
		f32v3 { 0.5f, 1.0f, -0.5f },
		f32v3 { -0.5f, 1.0f, 0.5f },
	};
	f32v3 const from { 0.0f, 0.0f, 0.0f };
	f32v3 const normal { 0.0f, 1.0f, 0.0f };
	f64 const area = triangle_area(tri);

	f32 const scales[] = { 0.01f, 1.0f, 100.0f };
	f64 normalizationAtUnitScale = 0.0;

	for (f32 const scale : scales) {
		CAPTURE(scale);
		ScopedScene const scene(
			make_scene({ tri }, { 1u }, f32v3 { scale, scale, scale })
		);
		FillStats const stats = (
			run_fill(
				pipeline, scene.scene, from, normal, /*hasNormal=*/true,
				200000u, 0x51ED0004u
			)
		);

		CHECK(stats.nonFiniteCount == 0u);
		f64 const relative = (stats.meanWeight - area) / area;
		f64 const relativeSigma = stats.stdErrWeight / area;
		CAPTURE(stats.meanWeight);
		CAPTURE(relative);
		CHECK(std::abs(relative) < std::max(5.0 * relativeSigma, 0.01));

		// and the normalization must scale exactly with it
		if (scale == 1.0f) {
			normalizationAtUnitScale = stats.meanNormalization;
		}
	}

	REQUIRE(normalizationAtUnitScale > 0.0);
}

// ---------------------------------------------------------------------------
// 10.4 -- degenerate inputs stay finite
//
// the fill loop runs on every live cell every frame, including cells whose
// representative point is somewhere useless. none of these may produce a nan or
// an infinity: W flows straight into radiance, and a single nan there
// propagates through accumulation and is very hard to trace back to this loop.
// ---------------------------------------------------------------------------

TEST_CASE("regir fill: degenerate configurations stay finite") {
	ScopedPipeline const pipeline;

	Tri const tri {
		f32v3 { -0.5f, 1.0f, -0.5f },
		f32v3 { 0.5f, 1.0f, -0.5f },
		f32v3 { -0.5f, 1.0f, 0.5f },
	};
	Tri const degenerate {
		f32v3 { 0.0f, 1.0f, 0.0f },
		f32v3 { 0.0f, 1.0f, 0.0f },
		f32v3 { 0.0f, 1.0f, 0.0f },
	};

	struct Case {
		char const * name;
		Tri tri;
		f32v3 from;
		f32v3 normal;
		bool hasNormal;
	};

	Case const cases[] = {
		// receiver facing away: every candidate scores 0, so every reservoir
		// finalizes to W == 0. legal and must not be a nan
		{ "facing away", tri, f32v3 { 0.0f, 0.0f, 0.0f },
			f32v3 { 0.0f, -1.0f, 0.0f }, true },
		// in the emitter's plane: cos at the light is 0 everywhere
		{ "coplanar", tri, f32v3 { 0.0f, 1.0f, 2.0f },
			f32v3 { 0.0f, 1.0f, 0.0f }, true },
		// at a vertex: distance 0 for part of the triangle
		{ "at a vertex", tri, f32v3 { -0.5f, 1.0f, -0.5f },
			f32v3 { 0.0f, 1.0f, 0.0f }, true },
		{ "zero area", degenerate, f32v3 { 0.0f, 0.0f, 0.0f },
			f32v3 { 0.0f, 1.0f, 0.0f }, true },
		{ "very close", tri, f32v3 { 0.0f, 0.999999f, 0.0f },
			f32v3 { 0.0f, 1.0f, 0.0f }, true },
		{ "very far", tri, f32v3 { 0.0f, -1e6f, 0.0f },
			f32v3 { 0.0f, 1.0f, 0.0f }, true },
		// the medium bucket: no normal, so the receiver cosine degrades to 1
		{ "no normal (medium)", tri, f32v3 { 0.0f, 0.0f, 0.0f },
			f32v3 { 0.0f, 0.0f, 0.0f }, false },
	};

	for (Case const & c : cases) {
		CAPTURE(c.name);
		ScopedScene const scene(make_scene({ c.tri }, { 1u }));
		FillStats const stats = (
			run_fill(
				pipeline, scene.scene, c.from, c.normal, c.hasNormal,
				4096u, 0x51ED0005u
			)
		);
		CHECK(stats.nonFiniteCount == 0u);
	}
}

// ---------------------------------------------------------------------------
// 10.5 -- a medium vertex is unbiased too
//
// the reserved normal bucket degrades the receiver cosine to 1 rather than to a
// dot product against a zero vector. a silent zero there would delete every
// light from every medium scatter, with no nan and no error -- the render would
// just be dark in fog. so the identity gets checked in that configuration on
// its own.
// ---------------------------------------------------------------------------

TEST_CASE("regir fill: E[W] equals the area at a normal-free vertex") {
	ScopedPipeline const pipeline;

	Tri const tri {
		f32v3 { -0.5f, 1.0f, -0.5f },
		f32v3 { 0.5f, 1.0f, -0.5f },
		f32v3 { -0.5f, 1.0f, 0.5f },
	};
	ScopedScene const scene(make_scene({ tri }, { 1u }));

	FillStats const stats = (
		run_fill(
			pipeline, scene.scene,
			f32v3 { 0.0f, 0.0f, 0.0f }, f32v3 { 0.0f, 0.0f, 0.0f },
			/*hasNormal=*/false, 200000u, 0x51ED0006u
		)
	);

	CHECK(stats.nonFiniteCount == 0u);
	REQUIRE(stats.validCount > 0u);

	f64 const area = triangle_area(tri);
	f64 const relative = (stats.meanWeight - area) / area;
	f64 const relativeSigma = stats.stdErrWeight / area;
	CAPTURE(area);
	CAPTURE(stats.meanWeight);
	CAPTURE(relative);
	CHECK(std::abs(relative) < std::max(5.0 * relativeSigma, 0.01));
}

// ---------------------------------------------------------------------------
// 10.6 -- reusing a cell away from its representative point
//
// this is the property regir actually rests on, and the one every earlier case
// here deliberately avoids: the reservoirs are built once at P_c, and then
// *every* shading point in the cell reuses them. that is only legitimate if
// reusing a proposal built somewhere else costs variance and not bias.
//
// it should be unbiased, and the reason is worth stating plainly, because it is
// the whole licence for the technique: W_Y estimates 1/p_true(Y) where p_true
// is the density the fill induced at P_c. the estimator f(p_shade, Y) * W_Y is
// then an importance-sampling estimate of the integral at p_shade using a
// proposal that happens to have been designed for P_c. any proposal whose
// support covers the integrand gives the right answer; a badly matched one only
// gives it more slowly.
//
// so the gate is a mean, and the *variance* is recorded rather than asserted --
// it is expected to grow with the offset, and watching it grow is how the cell
// size knob gets chosen later.
// ---------------------------------------------------------------------------

TEST_CASE("regir fill: a cell stays unbiased away from its representative") {
	ScopedPipeline const pipeline;
	ScopedDirectPipeline const directPipeline;

	Tri const tri {
		f32v3 { -0.5f, 2.0f, -0.5f },
		f32v3 { 0.5f, 2.0f, -0.5f },
		f32v3 { -0.5f, 2.0f, 0.5f },
	};
	f32v3 const representative { 0.0f, 0.0f, 0.0f };
	f32v3 const normal { 0.0f, 1.0f, 0.0f };
	ScopedScene const scene(make_scene({ tri }, { 1u }));

	// offsets in the receiver's tangent plane, in units of a plausible cell
	// size. 0 is the control: fill and shade at the same point
	f32 const offsets[] = { 0.0f, 0.125f, 0.25f, 0.5f };

	for (f32 const offset : offsets) {
		CAPTURE(offset);
		f32v3 const shading { offset, 0.0f, offset * 0.5f };

		DirectStats const stats = (
			run_direct(
				directPipeline, scene.scene, representative, normal,
				shading, normal, /*sampleCount=*/4096u,
				/*fillRealizations=*/512u, 0x51ED0007u
			)
		);

		// the same target integral, evaluated at the shading point: with a
		// single unoccluded emitter this is the exact irradiance
		f64 const reference = (
			target_integral(tri, shading, normal, /*hasNormal=*/true, 1.0, 96u)
		);
		f64 const relative = (stats.mean - reference) / reference;
		f64 const relativeSigma = stats.stdErr / reference;
		CAPTURE(reference);
		CAPTURE(stats.mean);
		CAPTURE(relative);
		CAPTURE(relativeSigma);
		MESSAGE(
			"offset " << offset
			<< " relative " << relative
			<< " relative sigma " << relativeSigma
		);

		CHECK(stats.nonFiniteCount == 0u);
		CHECK(std::abs(relative) < std::max(5.0 * relativeSigma, 0.01));
	}
}
