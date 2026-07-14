#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

struct FramePush {
	u64 inVa;
	u64 outVa;
	u32 count;
};

static constexpr u32 kLocalSize = 64;
static u32 groups_for(u32 count) { return (count + kLocalSize - 1) / kLocalSize; }

struct XyCrosscheck {
	f32v3 binormal, bitangent; // mine
	f32v3 b1, b2;              // paper's reference
};

std::vector<XyCrosscheck> evaluate_xy_crosscheck(std::vector<f32v3> const & samples) {
	u32 const count = (u32)samples.size();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "shading_frame_xy_crosscheck.comp",
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
	auto outBuf = vkof::buffer_create({
		.byteCount = count * 12u * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	FramePush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	auto flat = test::readback<f32>(outBuf, 0, count * 12u);

	std::vector<XyCrosscheck> out(count);
	for (u32 i = 0; i < count; ++i) {
		f32 const * p = flat.data() + i * 12u;
		out[i] = {
			{ p[0], p[1], p[2] }, { p[3], p[4], p[5] },
			{ p[6], p[7], p[8] }, { p[9], p[10], p[11] },
		};
	}

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

struct Frame { f32v3 tanX, tanY, nor; };

// each sample is 6 floats: nor.xyz, tangent.xyz
std::vector<Frame> evaluate_from_tangent(std::vector<f32> const & samplesFlat) {
	u32 const count = (u32)samplesFlat.size() / 6u;
	REQUIRE(samplesFlat.size() == (size_t)count * 6u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "shading_frame_from_tangent_evaluate.comp",
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
		.byteCount = count * 9u * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	FramePush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	auto flat = test::readback<f32>(outBuf, 0, count * 9u);

	std::vector<Frame> out(count);
	for (u32 i = 0; i < count; ++i) {
		f32 const * p = flat.data() + i * 9u;
		out[i] = { { p[0],p[1],p[2] }, { p[3],p[4],p[5] }, { p[6],p[7],p[8] } };
	}

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

// each sample is 10 floats: tanX.xyz, tanY.xyz, nor.xyz, rotation
std::vector<Frame> evaluate_rotate(std::vector<f32> const & samplesFlat) {
	u32 const count = (u32)samplesFlat.size() / 10u;
	REQUIRE(samplesFlat.size() == (size_t)count * 10u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "shading_frame_rotate_evaluate.comp",
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
		.byteCount = count * 9u * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	FramePush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	auto flat = test::readback<f32>(outBuf, 0, count * 9u);

	std::vector<Frame> out(count);
	for (u32 i = 0; i < count; ++i) {
		f32 const * p = flat.data() + i * 9u;
		out[i] = { { p[0],p[1],p[2] }, { p[3],p[4],p[5] }, { p[6],p[7],p[8] } };
	}

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

f32 dot3(f32v3 a, f32v3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
f32 len3(f32v3 a) { return std::sqrt(dot3(a, a)); }

f32v3 random_unit_vector(u32 & seed) {
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	f32 z = next() * 2.0f - 1.0f;
	f32 phi = next() * 6.28318530717958647692f;
	f32 r = std::sqrt(std::max(0.0f, 1.0f - z * z));
	return { r * std::cos(phi), r * std::sin(phi), z };
}

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("shading frame: utilCalculateXy golden value at nor=(0,0,1)") {
	std::vector<f32v3> const samples = { { 0.0f, 0.0f, 1.0f } };
	auto const result = evaluate_xy_crosscheck(samples);

	CHECK(result[0].binormal.x == doctest::Approx(1.0).epsilon(0.0001));
	CHECK(result[0].binormal.y == doctest::Approx(0.0).epsilon(0.0001));
	CHECK(result[0].binormal.z == doctest::Approx(0.0).epsilon(0.0001));
	CHECK(result[0].bitangent.x == doctest::Approx(0.0).epsilon(0.0001));
	CHECK(result[0].bitangent.y == doctest::Approx(1.0).epsilon(0.0001));
	CHECK(result[0].bitangent.z == doctest::Approx(0.0).epsilon(0.0001));
}

TEST_CASE("shading frame: utilCalculateXy orthonormal (fuzz, including both poles)") {
	std::vector<f32v3> samples;
	u32 seed = 11u;
	for (u32 i = 0; i < 2048u; ++i) { samples.push_back(random_unit_vector(seed)); }
	// explicit poles: the historically dangerous nor.z -> -1 case, and its
	// mirror nor.z -> 1
	samples.push_back({ 0.0f, 0.0f, 1.0f });
	samples.push_back({ 0.0f, 0.0f, -1.0f });
	samples.push_back({ 1e-6f, -1e-6f, -0.9999999f });
	samples.push_back({ -1e-6f, 1e-6f, 0.9999999f });

	auto const result = evaluate_xy_crosscheck(samples);
	for (u32 i = 0; i < result.size(); ++i) {
		f32v3 const & nor = samples[i];
		f32v3 const & u = result[i].binormal;
		f32v3 const & v = result[i].bitangent;
		CAPTURE(i);
		CHECK(len3(u) == doctest::Approx(1.0).epsilon(0.001));
		CHECK(len3(v) == doctest::Approx(1.0).epsilon(0.001));
		CHECK(dot3(nor, u) == doctest::Approx(0.0).epsilon(0.001));
		CHECK(dot3(nor, v) == doctest::Approx(0.0).epsilon(0.001));
		CHECK(dot3(u, v) == doctest::Approx(0.0).epsilon(0.001));
	}
}

TEST_CASE("shading frame: utilCalculateXy holds at the paper's adversarial vectors") {
	// Duff et al. 2017 JCGT, section 2: these two vectors are where the
	// ORIGINAL (non-revised) Frisvad method catastrophically fails (RMS
	// orthogonality error 0.29 and 0.16, one with flipped handedness).
	// cull implements the revised/branchless method, which the paper
	// reports at ~2e-8 RMS error even in the worst case -- confirm that
	// holds here, not the ~0.3 failure of the original method.
	std::vector<f32v3> const samples = {
		{ 0.00038527316f, 0.00038460016f, -0.99999988079f },
		{ -0.00019813581f, -0.00008946839f, -0.99999988079f },
	};
	auto const result = evaluate_xy_crosscheck(samples);

	for (u32 i = 0; i < result.size(); ++i) {
		f32v3 const & nor = samples[i];
		f32v3 const & u = result[i].binormal;
		f32v3 const & v = result[i].bitangent;
		CAPTURE(i);
		CHECK(len3(u) == doctest::Approx(1.0).epsilon(0.001));
		CHECK(len3(v) == doctest::Approx(1.0).epsilon(0.001));
		CHECK(dot3(nor, u) == doctest::Approx(0.0).epsilon(0.001));
		CHECK(dot3(nor, v) == doctest::Approx(0.0).epsilon(0.001));
		CHECK(dot3(u, v) == doctest::Approx(0.0).epsilon(0.001));
		// handedness: cross(nor, u) should point along v, not -v
		f32v3 const c = {
			nor.y*u.z - nor.z*u.y, nor.z*u.x - nor.x*u.z, nor.x*u.y - nor.y*u.x
		};
		CHECK(dot3(c, v) > 0.0f);
	}
}

TEST_CASE("shading frame: utilCalculateXy matches the paper's own branchlessONB") {
	std::vector<f32v3> samples;
	u32 seed = 22u;
	for (u32 i = 0; i < 2048u; ++i) { samples.push_back(random_unit_vector(seed)); }
	auto const result = evaluate_xy_crosscheck(samples);

	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(result[i].binormal.x == doctest::Approx(result[i].b1.x).epsilon(0.0001));
		CHECK(result[i].binormal.y == doctest::Approx(result[i].b1.y).epsilon(0.0001));
		CHECK(result[i].binormal.z == doctest::Approx(result[i].b1.z).epsilon(0.0001));
		CHECK(result[i].bitangent.x == doctest::Approx(result[i].b2.x).epsilon(0.0001));
		CHECK(result[i].bitangent.y == doctest::Approx(result[i].b2.y).epsilon(0.0001));
		CHECK(result[i].bitangent.z == doctest::Approx(result[i].b2.z).epsilon(0.0001));
	}
}

TEST_CASE("shading frame: fromTangent matches input tangent when already perpendicular") {
	std::vector<f32> flat;
	std::vector<f32v3> tangents;
	u32 seed = 33u;
	for (u32 i = 0; i < 256u; ++i) {
		f32v3 const nor = random_unit_vector(seed);
		// build a tangent guaranteed perpendicular to nor
		f32v3 arbitrary = random_unit_vector(seed);
		f32v3 t = {
			arbitrary.x - nor.x * dot3(nor, arbitrary),
			arbitrary.y - nor.y * dot3(nor, arbitrary),
			arbitrary.z - nor.z * dot3(nor, arbitrary),
		};
		f32 const l = len3(t);
		if (l < 1e-4f) { continue; }
		t = { t.x / l, t.y / l, t.z / l };
		flat.push_back(nor.x); flat.push_back(nor.y); flat.push_back(nor.z);
		flat.push_back(t.x); flat.push_back(t.y); flat.push_back(t.z);
		tangents.push_back(t);
	}
	auto const result = evaluate_from_tangent(flat);

	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(result[i].tanX.x == doctest::Approx(tangents[i].x).epsilon(0.001));
		CHECK(result[i].tanX.y == doctest::Approx(tangents[i].y).epsilon(0.001));
		CHECK(result[i].tanX.z == doctest::Approx(tangents[i].z).epsilon(0.001));
	}
}

TEST_CASE("shading frame: fromTangent orthogonalizes a non-perpendicular tangent") {
	std::vector<f32> flat;
	u32 seed = 44u;
	for (u32 i = 0; i < 512u; ++i) {
		f32v3 const nor = random_unit_vector(seed);
		f32v3 const tangent = random_unit_vector(seed); // not perpendicular in general
		flat.push_back(nor.x); flat.push_back(nor.y); flat.push_back(nor.z);
		flat.push_back(tangent.x); flat.push_back(tangent.y); flat.push_back(tangent.z);
	}
	auto const result = evaluate_from_tangent(flat);

	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(len3(result[i].tanX) == doctest::Approx(1.0).epsilon(0.001));
		CHECK(dot3(result[i].nor, result[i].tanX) == doctest::Approx(0.0).epsilon(0.001));
		CHECK(len3(result[i].tanY) == doctest::Approx(1.0).epsilon(0.001));
		CHECK(dot3(result[i].nor, result[i].tanY) == doctest::Approx(0.0).epsilon(0.001));
		CHECK(dot3(result[i].tanX, result[i].tanY) == doctest::Approx(0.0).epsilon(0.001));
	}
}

TEST_CASE("shading frame: fromTangent falls back to frisvad frame for a degenerate tangent") {
	std::vector<f32v3> const nors = {
		{ 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f },
	};

	std::vector<f32> flat;
	std::vector<f32v3> xyOnlySamples;
	for (f32v3 const & nor : nors) {
		flat.push_back(nor.x); flat.push_back(nor.y); flat.push_back(nor.z);
		flat.push_back(nor.x); flat.push_back(nor.y); flat.push_back(nor.z); // tangent == nor
		xyOnlySamples.push_back(nor);
	}
	auto const fromTangentResult = evaluate_from_tangent(flat);
	auto const xyResult = evaluate_xy_crosscheck(xyOnlySamples);

	for (u32 i = 0; i < fromTangentResult.size(); ++i) {
		CAPTURE(i);
		// shadingFrameFromNormal sets tanX=bitangent, tanY=binormal
		CHECK(fromTangentResult[i].tanX.x == doctest::Approx(xyResult[i].bitangent.x).epsilon(0.0001));
		CHECK(fromTangentResult[i].tanX.y == doctest::Approx(xyResult[i].bitangent.y).epsilon(0.0001));
		CHECK(fromTangentResult[i].tanX.z == doctest::Approx(xyResult[i].bitangent.z).epsilon(0.0001));
		CHECK(fromTangentResult[i].tanY.x == doctest::Approx(xyResult[i].binormal.x).epsilon(0.0001));
		CHECK(fromTangentResult[i].tanY.y == doctest::Approx(xyResult[i].binormal.y).epsilon(0.0001));
		CHECK(fromTangentResult[i].tanY.z == doctest::Approx(xyResult[i].binormal.z).epsilon(0.0001));
	}
}

TEST_CASE("shading frame: rotate by 0 returns the frame unchanged") {
	u32 seed = 55u;
	f32v3 const nor = random_unit_vector(seed);
	f32v3 binormal, bitangent;
	// build a valid frame via the crosscheck shader's "mine" path by reuse
	auto const xy = evaluate_xy_crosscheck({ nor });

	std::vector<f32> flat = {
		xy[0].bitangent.x, xy[0].bitangent.y, xy[0].bitangent.z,
		xy[0].binormal.x, xy[0].binormal.y, xy[0].binormal.z,
		nor.x, nor.y, nor.z,
		0.0f,
	};
	auto const result = evaluate_rotate(flat);

	CHECK(result[0].tanX.x == doctest::Approx(xy[0].bitangent.x).epsilon(0.0001));
	CHECK(result[0].tanY.x == doctest::Approx(xy[0].binormal.x).epsilon(0.0001));
	CHECK(result[0].nor.x == doctest::Approx(nor.x).epsilon(0.0001));
}

TEST_CASE("shading frame: rotate by pi/2 swaps tanX -> tanY, tanY -> -tanX") {
	u32 seed = 66u;
	f32v3 const nor = random_unit_vector(seed);
	auto const xy = evaluate_xy_crosscheck({ nor });
	f32v3 const tanX = xy[0].bitangent, tanY = xy[0].binormal;

	constexpr f32 kHalfPi = 1.57079632679489661923f;
	std::vector<f32> const flat = {
		tanX.x, tanX.y, tanX.z, tanY.x, tanY.y, tanY.z, nor.x, nor.y, nor.z, kHalfPi,
	};
	auto const result = evaluate_rotate(flat);

	CHECK(result[0].tanX.x == doctest::Approx(tanY.x).epsilon(0.001));
	CHECK(result[0].tanX.y == doctest::Approx(tanY.y).epsilon(0.001));
	CHECK(result[0].tanX.z == doctest::Approx(tanY.z).epsilon(0.001));
	CHECK(result[0].tanY.x == doctest::Approx(-tanX.x).epsilon(0.001));
	CHECK(result[0].tanY.y == doctest::Approx(-tanX.y).epsilon(0.001));
	CHECK(result[0].tanY.z == doctest::Approx(-tanX.z).epsilon(0.001));
}

TEST_CASE("shading frame: rotate is orthonormality-preserving and composable (fuzz)") {
	u32 seed = 77u;
	std::vector<f32> singleFlat, composedFlat;
	std::vector<f32> angleAs, angleBs;
	for (u32 i = 0; i < 512u; ++i) {
		f32v3 const nor = random_unit_vector(seed);
		auto const xy = evaluate_xy_crosscheck({ nor });
		f32v3 const tanX = xy[0].bitangent, tanY = xy[0].binormal;

		seed = seed * 747796405u + 2891336453u;
		f32 const angleA = ((f32)(seed >> 8) / (f32)(1u << 24)) * 6.28318530717958647692f;
		seed = seed * 747796405u + 2891336453u;
		f32 const angleB = ((f32)(seed >> 8) / (f32)(1u << 24)) * 6.28318530717958647692f;

		singleFlat.insert(singleFlat.end(), {
			tanX.x, tanX.y, tanX.z, tanY.x, tanY.y, tanY.z, nor.x, nor.y, nor.z,
			angleA + angleB,
		});
		angleAs.push_back(angleA);
		angleBs.push_back(angleB);
	}
	auto const direct = evaluate_rotate(singleFlat);

	// now rotate by A, then rotate the result by B, and compare
	std::vector<f32> firstFlat;
	for (u32 i = 0; i < 512u; ++i) {
		f32 const * base = singleFlat.data() + i * 10u;
		firstFlat.insert(firstFlat.end(), {
			base[0], base[1], base[2], base[3], base[4], base[5],
			base[6], base[7], base[8], angleAs[i],
		});
	}
	auto const afterA = evaluate_rotate(firstFlat);

	std::vector<f32> secondFlat;
	for (u32 i = 0; i < 512u; ++i) {
		secondFlat.insert(secondFlat.end(), {
			afterA[i].tanX.x, afterA[i].tanX.y, afterA[i].tanX.z,
			afterA[i].tanY.x, afterA[i].tanY.y, afterA[i].tanY.z,
			afterA[i].nor.x, afterA[i].nor.y, afterA[i].nor.z,
			angleBs[i],
		});
	}
	auto const composed = evaluate_rotate(secondFlat);

	for (u32 i = 0; i < direct.size(); ++i) {
		CAPTURE(i);
		// orthonormality preserved
		CHECK(len3(direct[i].tanX) == doctest::Approx(1.0).epsilon(0.001));
		CHECK(len3(direct[i].tanY) == doctest::Approx(1.0).epsilon(0.001));
		CHECK(dot3(direct[i].tanX, direct[i].tanY) == doctest::Approx(0.0).epsilon(0.001));
		CHECK(dot3(direct[i].nor, direct[i].tanX) == doctest::Approx(0.0).epsilon(0.001));
		// composability: rotate(A+B) == rotate(rotate(A), B)
		CHECK(direct[i].tanX.x == doctest::Approx(composed[i].tanX.x).epsilon(0.001));
		CHECK(direct[i].tanX.y == doctest::Approx(composed[i].tanX.y).epsilon(0.001));
		CHECK(direct[i].tanX.z == doctest::Approx(composed[i].tanX.z).epsilon(0.001));
		CHECK(direct[i].tanY.x == doctest::Approx(composed[i].tanY.x).epsilon(0.001));
		CHECK(direct[i].tanY.y == doctest::Approx(composed[i].tanY.y).epsilon(0.001));
		CHECK(direct[i].tanY.z == doctest::Approx(composed[i].tanY.z).epsilon(0.001));
	}
}

TEST_CASE("shading frame: rotate leaves nor unchanged") {
	u32 seed = 88u;
	std::vector<f32> flat;
	std::vector<f32v3> nors;
	for (u32 i = 0; i < 256u; ++i) {
		f32v3 const nor = random_unit_vector(seed);
		auto const xy = evaluate_xy_crosscheck({ nor });
		seed = seed * 747796405u + 2891336453u;
		f32 const angle = ((f32)(seed >> 8) / (f32)(1u << 24)) * 6.28318530717958647692f;
		flat.insert(flat.end(), {
			xy[0].bitangent.x, xy[0].bitangent.y, xy[0].bitangent.z,
			xy[0].binormal.x, xy[0].binormal.y, xy[0].binormal.z,
			nor.x, nor.y, nor.z, angle,
		});
		nors.push_back(nor);
	}
	auto const result = evaluate_rotate(flat);
	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(result[i].nor.x == doctest::Approx(nors[i].x).epsilon(0.0001));
		CHECK(result[i].nor.y == doctest::Approx(nors[i].y).epsilon(0.0001));
		CHECK(result[i].nor.z == doctest::Approx(nors[i].z).epsilon(0.0001));
	}
}

TEST_CASE("shading frame: orthogonality error heatmap over the southern hemisphere") {
	// reproduces the validation methodology of the paper's own figure 1:
	// unit vectors (x, y, -sqrt(1-x^2-y^2)) over the unit disc, colored by
	// rms deviation from orthogonality. the paper's original (non-revised)
	// frisvad method shows a bright error spike at the disc's center
	// (nor.z -> -1); the revised method cull uses should not.
	constexpr u32 kSize = 512;

	std::vector<f32v3> samples(kSize * kSize);
	std::vector<bool> valid(kSize * kSize, false);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const ny = 2.0f * (f32)y / (f32)(kSize - 1) - 1.0f;
		for (u32 x = 0; x < kSize; ++x) {
			f32 const nx = 2.0f * (f32)x / (f32)(kSize - 1) - 1.0f;
			f32 const r2 = nx * nx + ny * ny;
			u32 const idx = y * kSize + x;
			if (r2 <= 1.0f) {
				valid[idx] = true;
				samples[idx] = { nx, ny, -std::sqrt(1.0f - r2) };
			} else {
				samples[idx] = { 0.0f, 0.0f, -1.0f }; // dummy, masked out below
			}
		}
	}
	auto const result = evaluate_xy_crosscheck(samples);

	std::vector<f32> logErr(samples.size(), 0.0f);
	f32 maxLogErr = -1e9f;
	for (u32 i = 0; i < samples.size(); ++i) {
		if (!valid[i]) { continue; }
		f32v3 const & nor = samples[i];
		f32v3 const & u = result[i].binormal;
		f32v3 const & v = result[i].bitangent;
		f32 const eLenN = len3(nor) - 1.0f;
		f32 const eLenU = len3(u) - 1.0f;
		f32 const eLenV = len3(v) - 1.0f;
		f32 const eNU = dot3(nor, u);
		f32 const eNV = dot3(nor, v);
		f32 const eUV = dot3(u, v);
		f32 const mse = (
			eLenN*eLenN + eLenU*eLenU + eLenV*eLenV + eNU*eNU + eNV*eNV + eUV*eUV
		) / 6.0f;
		f32 const rms = std::sqrt(mse);
		// log scale, matching the paper's figure 1 (1e-9 .. 1e-1)
		f32 const l = std::log10(std::max(rms, 1e-9f));
		logErr[i] = l;
		maxLogErr = std::max(maxLogErr, l);
	}

	std::vector<f32> tonemapped(samples.size(), 0.0f);
	for (u32 i = 0; i < samples.size(); ++i) {
		if (!valid[i]) { continue; }
		// map [-9, -1] log-error range to [0, 1]
		tonemapped[i] = std::clamp((logErr[i] + 9.0f) / 8.0f, 0.0f, 1.0f);
	}

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		tonemapped, tonemapped, tonemapped, kSize, kSize,
		SHADING_FRAME_OUTPUT_DIR "shading_frame_orthogonality_error.png", &nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);
	CAPTURE(maxLogErr);
	// the paper reports worst-case rms error ~1e-7 (log10 ~ -7) for the
	// revised method across a billion samples; generous margin above that
	CHECK(maxLogErr < -4.0f);
}

} // TEST_SUITE("[headless]")
