#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

// tests for the openpbr conductor (metal) lobe,
// openPbrFresnelMetallicEvaluateF: f82-tint fresnel (kutz-hasan-edmondson
// 2021) x anisotropic ggx D x height-correlated smith V x turquin 2019
// conductor energy compensation. verified byte-identical against cull's
// util-material-openpbr-metallic.glsl -- no lib code changed for this
// milestone. the f82 fresnel *function* (utilFresnelMetallic) and the
// D/V/directional-albedo building blocks were each verified in earlier
// milestones (test-fresnel.cpp, test-microfacet.cpp,
// test-energy-compensation.cpp, all with their own materialx crosschecks);
// these tests target the assembled lobe: golden values against a cpu f64
// transcription, the f82 edge-dip showing through the full lobe, energy
// conservation under the turquin compensation, the lobe's (by-design,
// incident-side) non-reciprocity, and a full-lobe materialx crosscheck on
// mirror configurations.
//
// conventions pinned throughout: anisotropy rotation 0, coat off,
// thin-film off (thinFilmWeight = 0), normal = (0,0,1).

namespace {

static constexpr u32 kLocalSize = 64;
static u32 groups_for(u32 count) { return (count + kLocalSize - 1) / kLocalSize; }

constexpr f64 kPi = 3.14159265358979323846;

// gold-ish f0 used across visual tests
constexpr f32v3 skGoldColor = { 1.0f, 0.766f, 0.336f };

// -----------------------------------------------------------------------------
// cpu f64 reference, transcribed from the same formulas the shader ports
// (util-material-openpbr-metallic.glsl and the microfacet/albedo helpers
// it calls). not independent of the implementation -- it validates the
// gpu f32 evaluation against the intended math at double precision;
// independence comes from the materialx crosscheck below.
// -----------------------------------------------------------------------------

struct RefDir {
	f64 x;
	f64 y;
	f64 z;
};

f64 ref_roughness_alpha_t(f64 roughness, f64 anisotropy) {
	f64 const oneMinusA = 1.0 - anisotropy;
	return std::max(
		roughness * std::sqrt(2.0 / (1.0 + oneMinusA * oneMinusA)), 1e-5
	);
}

f64 ref_roughness_alpha_b(f64 roughness, f64 anisotropy) {
	f64 const oneMinusA = 1.0 - anisotropy;
	return std::max(oneMinusA * ref_roughness_alpha_t(roughness, anisotropy), 1e-5);
}

f64 ref_ggx_d_aniso(RefDir h, f64 ax, f64 ay) {
	f64 const d = std::max(
		h.x * h.x / (ax * ax) + h.y * h.y / (ay * ay) + h.z * h.z, 1e-7
	);
	return 1.0 / (kPi * ax * ay * d * d);
}

f64 ref_smith_v_aniso(RefDir wi, RefDir wo, f64 ax, f64 ay) {
	if (wi.z <= 0.0 || wo.z <= 0.0) { return 0.0; }
	f64 const aO = std::sqrt(
		ax * ax * wo.x * wo.x + ay * ay * wo.y * wo.y + wo.z * wo.z
	);
	f64 const aI = std::sqrt(
		ax * ax * wi.x * wi.x + ay * ay * wi.y * wi.y + wi.z * wi.z
	);
	return 0.5 / std::max(wi.z * aO + wo.z * aI, 1e-7);
}

// the materialx rational fit (identical constants in
// util-energy-compensation.glsl / util-material-openpbr-microfacet.glsl)
f64 ref_ggx_dir_albedo(f64 mu, f64 alpha, f64 f0, f64 f90) {
	f64 const x = mu;
	f64 const y = alpha;
	f64 const x2 = x * x;
	f64 const y2 = y * y;
	f64 r[4];
	f64 const c0[4] = { 0.1003, 0.9345, 1.0, 1.0 };
	f64 const c1[4] = { -0.6303, -2.323, -1.765, 0.2281 };
	f64 const c2[4] = { 9.748, 2.229, 8.263, 15.94 };
	f64 const c3[4] = { -2.038, -3.748, 11.53, -55.83 };
	f64 const c4[4] = { 29.34, 1.424, 28.96, 13.08 };
	f64 const c5[4] = { -8.245, -0.7684, -7.507, 41.26 };
	f64 const c6[4] = { -26.44, 1.436, -36.11, 54.9 };
	f64 const c7[4] = { 19.99, 0.2913, 15.86, 300.2 };
	f64 const c8[4] = { -5.448, 0.6286, 33.37, -285.1 };
	for (u32 k = 0; k < 4; ++k) {
		r[k] = (
			c0[k] + c1[k] * x + c2[k] * y + c3[k] * x * y + c4[k] * x2
			+ c5[k] * y2 + c6[k] * x2 * y + c7[k] * x * y2 + c8[k] * x2 * y2
		);
	}
	f64 const a = std::clamp(r[0] / r[2], 0.0, 1.0);
	f64 const b = std::clamp(r[1] / r[3], 0.0, 1.0);
	return f0 * a + f90 * b;
}

// one color channel of the conductor lobe
f64 ref_metallic_f(
	RefDir wi, RefDir wo,
	f64 baseWeight, f64 baseColorC, f64 specularWeight, f64 specularColorC,
	f64 roughness, f64 anisotropy
) {
	f64 const dotNorWi = std::max(wi.z, 1e-5);
	f64 const f0 = baseWeight * baseColorC;

	f64 const fresnelConstS = f0 + (1.0 - f0) * 0.462664;
	f64 const fresnelConst = (
		fresnelConstS * (1.0 - specularWeight * specularColorC)
	);
	f64 const fresnelDenom = 0.056653;
	f64 const fs = f0 + (1.0 - f0) * std::pow(1.0 - dotNorWi, 5.0);
	// clamped at zero, matching the production lobe's deliberate
	// deviation from cull (user decision 2026-07-14)
	f64 const f82 = std::max(
		fs
		- (
			(dotNorWi * std::pow(1.0 - dotNorWi, 6.0) / fresnelDenom)
			* fresnelConst
		),
		0.0
	);

	f64 const ax = ref_roughness_alpha_t(roughness, anisotropy);
	f64 const ay = ref_roughness_alpha_b(roughness, anisotropy);
	RefDir const h0 = { wi.x + wo.x, wi.y + wo.y, wi.z + wo.z };
	f64 const hLen = std::sqrt(h0.x * h0.x + h0.y * h0.y + h0.z * h0.z);
	RefDir const h = { h0.x / hLen, h0.y / hLen, h0.z / hLen };
	f64 const d = ref_ggx_d_aniso(h, ax, ay);
	f64 const v = ref_smith_v_aniso(wi, wo, ax, ay);

	f64 const fAvg = f0 + (1.0 - f0) * (1.0 / 21.0);
	f64 const ess = std::max(
		ref_ggx_dir_albedo(dotNorWi, std::sqrt(ax * ay), 1.0, 1.0), 1e-4
	);
	f64 const comp = 1.0 + fAvg * (1.0 - ess) / ess;

	return f82 * d * v * comp;
}

// -----------------------------------------------------------------------------
// gpu dispatch helpers
// -----------------------------------------------------------------------------

// metallic_lobe_evaluate.comp: 14 floats/sample
void push_metal_sample(
	std::vector<f32> & flat,
	f32v3 wi, f32v3 wo,
	f32 baseWeight, f32v3 baseColor,
	f32 specularWeight, f32 specularColor,
	f32 specularRoughness, f32 specularRoughnessAnisotropy
) {
	flat.push_back(wi.x); flat.push_back(wi.y); flat.push_back(wi.z);
	flat.push_back(wo.x); flat.push_back(wo.y); flat.push_back(wo.z);
	flat.push_back(baseWeight);
	flat.push_back(baseColor.x); flat.push_back(baseColor.y); flat.push_back(baseColor.z);
	flat.push_back(specularWeight);
	flat.push_back(specularColor);
	flat.push_back(specularRoughness);
	flat.push_back(specularRoughnessAnisotropy);
}

std::vector<f32v3> evaluate_metal(std::vector<f32> const & samplesFlat) {
	u32 const count = (u32)samplesFlat.size() / 14u;
	REQUIRE(samplesFlat.size() == (size_t)count * 14u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "metallic_lobe_evaluate.comp",
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

// metallic_lobe_furnace_integrate.comp: 8 floats/sample
void push_furnace_sample(
	std::vector<f32> & flat,
	f32 muI, f32 specularRoughness, f32 specularRoughnessAnisotropy,
	f32 baseWeight, f32 baseColorScalar, f32 specularWeight,
	f32 specularColorScalar
) {
	flat.push_back(muI);
	flat.push_back(specularRoughness);
	flat.push_back(specularRoughnessAnisotropy);
	flat.push_back(baseWeight);
	flat.push_back(baseColorScalar);
	flat.push_back(specularWeight);
	flat.push_back(specularColorScalar);
	flat.push_back(0.0f);
}

std::vector<f32> integrate_metal_furnace(
	std::vector<f32> const & samplesFlat, u32 thetaSteps, u32 phiSteps
) {
	u32 const count = (u32)samplesFlat.size() / 8u;
	REQUIRE(samplesFlat.size() == (size_t)count * 8u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "metallic_lobe_furnace_integrate.comp",
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

// metallic_lobe_crosscheck.comp: 9 floats/sample, mirror configurations
void push_crosscheck_sample(
	std::vector<f32> & flat,
	f32 mu, f32 specularRoughness,
	f32 baseWeight, f32v3 baseColor,
	f32 specularWeight, f32 specularColor
) {
	flat.push_back(mu);
	flat.push_back(specularRoughness);
	flat.push_back(baseWeight);
	flat.push_back(baseColor.x); flat.push_back(baseColor.y); flat.push_back(baseColor.z);
	flat.push_back(specularWeight);
	flat.push_back(specularColor);
	flat.push_back(0.0f);
}

void crosscheck_metal(
	std::vector<f32> const & samplesFlat,
	std::vector<f32v3> & outMine,
	std::vector<f32v3> & outRef
) {
	u32 const count = (u32)samplesFlat.size() / 9u;
	REQUIRE(samplesFlat.size() == (size_t)count * 9u);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "metallic_lobe_crosscheck.comp",
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
	auto mineBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v3),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	auto refBuf = vkof::buffer_create({
		.byteCount = count * sizeof(f32v3),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push { u64 inVa; u64 mineOutVa; u64 refOutVa; u32 count; };
	Push const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.mineOutVa = vkof::buffer_virtual_address(mineBuf),
		.refOutVa = vkof::buffer_virtual_address(refBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups_for(count));

	outMine = test::readback<f32v3>(mineBuf, 0, count);
	outRef = test::readback<f32v3>(refBuf, 0, count);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(mineBuf);
	vkof::buffer_destroy(refBuf);
	vkof::pipeline_destroy(pl);
}

f32v3 dir_from(f32 cosTheta, f32 phi) {
	f32 const sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
	return { sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta };
}

} // namespace

TEST_SUITE("[headless]") {

// -----------------------------------------------------------------------------
// golden values
// -----------------------------------------------------------------------------

TEST_CASE("metallic lobe: golden values at normal incidence (wi == wo == n)") {
	// at mu = 1: fs = f0 exactly (schlick term vanishes), the f82
	// correction term is 0 (mu (1-mu)^6 = 0), D = 1/(pi alpha^2),
	// V = 0.25, so f = f0 * D * 0.25 * comp
	std::vector<f32> const roughnesses = { 0.05f, 0.2f, 0.5f, 1.0f };
	std::vector<f32> const grays = { 0.2f, 0.6f, 1.0f };

	std::vector<f32> flat;
	std::vector<f64> expected;
	for (f32 rough : roughnesses) {
		for (f32 g : grays) {
			push_metal_sample(
				flat, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 1.0f },
				1.0f, { g, g, g }, 1.0f, 1.0f, rough, 0.0f
			);
			expected.push_back(
				ref_metallic_f(
					{ 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0 },
					1.0, g, 1.0, 1.0, rough, 0.0
				)
			);
		}
	}
	auto const result = evaluate_metal(flat);

	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(result[i].x == doctest::Approx(expected[i]).epsilon(0.002));
	}
}

TEST_CASE("metallic lobe: f82 edge dip shows through the full lobe at mu = 1/7") {
	// F82(1/7) = Fs(1/7) * Sw * Sc by construction; D, V and the
	// compensation are independent of specularColor, so the ratio of two
	// otherwise identical lobes with different specularColor equals the
	// specularColor ratio exactly at that one cosine
	f32 const mu = 1.0f / 7.0f;
	std::vector<f32> const scs = { 0.25f, 0.5f, 0.75f };
	f32 const rough = 0.3f;

	std::vector<f32> flat;
	f32v3 const wi = dir_from(mu, 0.0f);
	f32v3 const wo = { -wi.x, -wi.y, wi.z };
	// reference lobe at specularColor = 1
	push_metal_sample(
		flat, wi, wo, 1.0f, { 1.0f, 1.0f, 1.0f }, 1.0f, 1.0f, rough, 0.0f
	);
	for (f32 sc : scs) {
		push_metal_sample(
			flat, wi, wo, 1.0f, { 1.0f, 1.0f, 1.0f }, 1.0f, sc, rough, 0.0f
		);
	}
	auto const result = evaluate_metal(flat);

	// f0 = 1 makes Fs(mu) = 1 for every mu, so lobe(sc)/lobe(1) at
	// mu = 1/7 is exactly sc
	for (u32 i = 0; i < scs.size(); ++i) {
		CAPTURE(scs[i]);
		CHECK(
			result[i + 1].x / result[0].x
			== doctest::Approx(scs[i]).epsilon(0.002)
		);
	}
}

TEST_CASE("metallic lobe: matches cpu f64 reference over a dense grid (iso + aniso)") {
	std::vector<f32> const roughnesses = { 0.05f, 0.3f, 0.7f, 1.0f };
	std::vector<f32> const anisos = { 0.0f, 0.5f, 0.9f };
	std::vector<f32> const cosines = { 0.05f, 0.3f, 0.7f, 1.0f };
	std::vector<f32> const phis = { 0.3f, 2.0f, 4.4f };
	f32v3 const color = { 0.95f, 0.5f, 0.2f };

	std::vector<f32> flat;
	std::vector<f64> expectedX, expectedZ;
	for (f32 rough : roughnesses) {
		for (f32 aniso : anisos) {
			for (f32 muI : cosines) {
				for (f32 muO : cosines) {
					for (f32 phi : phis) {
						f32v3 const wi = dir_from(muI, 0.0f);
						f32v3 const wo = dir_from(muO, phi);
						push_metal_sample(
							flat, wi, wo, 1.0f, color, 1.0f, 0.8f, rough, aniso
						);
						RefDir const rwi = { wi.x, wi.y, wi.z };
						RefDir const rwo = { wo.x, wo.y, wo.z };
						expectedX.push_back(
							ref_metallic_f(
								rwi, rwo, 1.0, color.x, 1.0, 0.8, rough, aniso
							)
						);
						expectedZ.push_back(
							ref_metallic_f(
								rwi, rwo, 1.0, color.z, 1.0, 0.8, rough, aniso
							)
						);
					}
				}
			}
		}
	}
	auto const result = evaluate_metal(flat);

	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(result[i].x == doctest::Approx(expectedX[i]).epsilon(0.005));
		CHECK(result[i].z == doctest::Approx(expectedZ[i]).epsilon(0.005));
	}
}

// -----------------------------------------------------------------------------
// physical properties
// -----------------------------------------------------------------------------

TEST_CASE("metallic lobe: non-negative and finite over the full parameter domain (fuzz)") {
	// full-domain fuzz: with the f82 clamp in place (see the regression
	// test below) the lobe must be non-negative everywhere, including
	// the dark-f0 / low-specular-weight region where the raw model dips
	// negative
	std::vector<f32> flat;
	u32 seed = 777u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 4096u; ++i) {
		f32 const muI = (i % 5u == 0u) ? next() * 0.01f : next();
		f32 const muO = (i % 5u == 1u) ? next() * 0.01f : next();
		push_metal_sample(
			flat,
			dir_from(muI, next() * 6.2831853f),
			dir_from(muO, next() * 6.2831853f),
			next(), { next(), next(), next() },
			next(), next(), 0.02f + next() * 0.98f, next()
		);
	}
	auto const result = evaluate_metal(flat);

	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CHECK(result[i].x >= -1e-5f);
		CHECK(std::isfinite(result[i].x));
		CHECK(std::isfinite(result[i].y));
		CHECK(std::isfinite(result[i].z));
	}
}

TEST_CASE("metallic lobe: f82 clamp holds at the documented negative point (regression)") {
	// the raw f82 correction F82(mu) = Fs(mu) - w(mu) Fc dips below zero
	// for small f0 AND small specularWeight*specularColor (analytically:
	// f0 = 0, SwSc = 0 gives raw F82(0.3) ~= -0.12; this exact config
	// measured -0.416 on the assembled lobe before the clamp). the
	// production lobe now clamps f82 at zero -- a deliberate deviation
	// from cull, user decision 2026-07-14. this anchors that clamp: the
	// lobe here must be exactly zero, not negative
	std::vector<f32> flat;
	f32 const mu = 0.3f;
	f32v3 const wi = dir_from(mu, 0.0f);
	f32v3 const wo = { -wi.x, -wi.y, wi.z };
	push_metal_sample(
		flat, wi, wo, /*baseWeight*/ 0.0f, { 0.0f, 0.0f, 0.0f },
		/*specularWeight*/ 0.0f, /*specularColor*/ 0.0f, 0.4f, 0.0f
	);
	auto const result = evaluate_metal(flat);

	CHECK(std::isfinite(result[0].x));
	MESSAGE("lobe value at the formerly-negative point: ", result[0].x);
	CHECK(result[0].x == 0.0f);
	CHECK(result[0].y == 0.0f);
	CHECK(result[0].z == 0.0f);
}

TEST_CASE("metallic lobe: exactly reciprocal when muI == muO (fuzz)") {
	// fresnel and compensation are functions of dot(nor, wi) only; D and
	// V are symmetric under swap, so the lobe is reciprocal exactly when
	// the two directions share a cosine
	std::vector<f32> forward, swapped;
	u32 seed = 31u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 512u; ++i) {
		f32 const mu = 0.05f + next() * 0.9f;
		f32v3 const a = dir_from(mu, next() * 6.2831853f);
		f32v3 const b = dir_from(mu, next() * 6.2831853f);
		f32 const rough = 0.02f + next() * 0.98f;
		f32 const aniso = next();
		f32v3 const color = { next(), next(), next() };
		push_metal_sample(forward, a, b, 1.0f, color, 1.0f, 0.7f, rough, aniso);
		push_metal_sample(swapped, b, a, 1.0f, color, 1.0f, 0.7f, rough, aniso);
	}
	auto const fwd = evaluate_metal(forward);
	auto const swp = evaluate_metal(swapped);

	for (u32 i = 0; i < fwd.size(); ++i) {
		CAPTURE(i);
		CHECK(fwd[i].x == doctest::Approx(swp[i].x).epsilon(0.002));
		CHECK(fwd[i].y == doctest::Approx(swp[i].y).epsilon(0.002));
		CHECK(fwd[i].z == doctest::Approx(swp[i].z).epsilon(0.002));
	}
}

TEST_CASE("metallic lobe: not reciprocal when muI != muO, by design") {
	// a regression that made this exactly reciprocal everywhere would
	// mean the fresnel/compensation silently stopped depending on the
	// incident cosine -- catch that
	std::vector<f32> forward, swapped;
	for (f32 muI : { 0.1f, 0.4f, 0.9f }) {
		for (f32 muO : { 0.1f, 0.4f, 0.9f }) {
			if (muI == muO) { continue; }
			f32v3 const a = dir_from(muI, 0.0f);
			f32v3 const b = dir_from(muO, 2.2f);
			push_metal_sample(
				forward, a, b, 1.0f, { 0.5f, 0.5f, 0.5f }, 1.0f, 0.8f, 0.4f, 0.0f
			);
			push_metal_sample(
				swapped, b, a, 1.0f, { 0.5f, 0.5f, 0.5f }, 1.0f, 0.8f, 0.4f, 0.0f
			);
		}
	}
	auto const fwd = evaluate_metal(forward);
	auto const swp = evaluate_metal(swapped);

	f64 maxRelDiff = 0.0;
	for (u32 i = 0; i < fwd.size(); ++i) {
		f64 const rel = (
			std::abs((f64)fwd[i].x - (f64)swp[i].x)
			/ std::max((f64)fwd[i].x, 1e-9)
		);
		maxRelDiff = std::max(maxRelDiff, rel);
	}
	MESSAGE("max relative |f(a,b)-f(b,a)|/f across mismatched-mu pairs: ", maxRelDiff);
	CHECK(maxRelDiff > 0.001);
}

// -----------------------------------------------------------------------------
// furnace: the turquin compensation's defining property. fixed wi
// (fresnel + compensation factor out), integrate wo:
//   R(wi) = F82(muI) * comp(muI) * E_actual(muI) ~= F82(muI)
// for a fully white metal (f0 = 1 -> F82 == 1, fAvg = 1) this is 1 at
// every incidence and roughness, to within the albedo fit's own accuracy
// -----------------------------------------------------------------------------

TEST_CASE("metallic lobe: white furnace == 1 across roughness x incidence") {
	// roughness floor 0.1: below that the near-mirror lobe is only a
	// couple of quadrature samples wide at 200x400 steps and the integral
	// itself is the error source, not the lobe (the dielectric-specular
	// furnace test starts at 0.1 for the same reason)
	std::vector<f32> const mus = { 0.05f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f };
	std::vector<f32> const roughnesses = { 0.1f, 0.3f, 0.6f, 1.0f };

	std::vector<f32> flat;
	std::vector<f32v2> params;
	for (f32 mu : mus) {
		for (f32 rough : roughnesses) {
			push_furnace_sample(flat, mu, rough, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f);
			params.push_back({ mu, rough });
		}
	}
	auto const result = integrate_metal_furnace(flat, 200u, 400u);

	f64 worst = 0.0;
	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(params[i].x);
		CAPTURE(params[i].y);
		worst = std::max(worst, std::abs((f64)result[i] - 1.0));
		// the fit's weakest corner is low roughness + grazing incidence,
		// same as every other consumer of this albedo fit
		f32 const tolerance = (
			(params[i].x <= 0.05f && params[i].y <= 0.1f) ? 0.08f : 0.03f
		);
		CHECK(result[i] == doctest::Approx(1.0).epsilon(tolerance));
	}
	MESSAGE("worst |furnace - 1| across roughness x incidence: ", worst);
}

TEST_CASE("metallic lobe: white furnace == 1 across anisotropy") {
	std::vector<f32> const anisos = { 0.0f, 0.25f, 0.5f, 0.75f, 0.95f };
	std::vector<f32> const mus = { 0.1f, 0.4f, 0.8f };
	f32 const rough = 0.35f;

	std::vector<f32> flat;
	std::vector<f32v2> params;
	for (f32 aniso : anisos) {
		for (f32 mu : mus) {
			push_furnace_sample(flat, mu, rough, aniso, 1.0f, 1.0f, 1.0f, 1.0f);
			params.push_back({ aniso, mu });
		}
	}
	auto const result = integrate_metal_furnace(flat, 200u, 400u);

	f64 worst = 0.0;
	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(params[i].x);
		CAPTURE(params[i].y);
		worst = std::max(worst, std::abs((f64)result[i] - 1.0));
		// the isotropic fit is indexed by sqrt(ax*ay); widen toward the
		// high-anisotropy end where that collapse is least faithful (the
		// needle-lobe quadrature also starts to strain there)
		f32 tolerance = 0.03f;
		if (params[i].x >= 0.9f) { tolerance = 0.2f; }
		else if (params[i].x >= 0.7f) { tolerance = 0.08f; }
		CHECK(result[i] == doctest::Approx(1.0).epsilon(tolerance));
	}
	MESSAGE("worst |furnace - 1| across anisotropy: ", worst);
}

TEST_CASE("metallic lobe: colored furnace matches F82(muI)-scaled prediction and stays under 1") {
	// for f0 < 1 the same factoring gives
	//   R(muI) ~= F82(muI) * (E + fAvg (1 - E)), E = E_fit(muI, alpha)
	// -- a per-sample closed-form prediction. also: R must never exceed 1
	std::vector<f32> const grays = { 0.2f, 0.5f, 0.8f };
	std::vector<f32> const mus = { 0.1f, 0.5f, 0.9f };
	f32 const rough = 0.4f;

	std::vector<f32> flat;
	std::vector<f64> predicted;
	for (f32 g : grays) {
		for (f32 mu : mus) {
			push_furnace_sample(flat, mu, rough, 0.0f, 1.0f, g, 1.0f, 1.0f);
			f64 const f0 = g;
			f64 const dotNorWi = mu;
			f64 const fresnelConstS = f0 + (1.0 - f0) * 0.462664;
			f64 const fs = f0 + (1.0 - f0) * std::pow(1.0 - dotNorWi, 5.0);
			f64 const f82 = (
				fs
				- (
					(dotNorWi * std::pow(1.0 - dotNorWi, 6.0) / 0.056653)
					* (fresnelConstS * (1.0 - 1.0))
				)
			);
			f64 const e = std::max(
				ref_ggx_dir_albedo(mu, rough, 1.0, 1.0), 1e-4
			);
			f64 const fAvg = f0 + (1.0 - f0) / 21.0;
			predicted.push_back(f82 * (e + fAvg * (1.0 - e)));
		}
	}
	auto const result = integrate_metal_furnace(flat, 200u, 400u);

	f64 worst = 0.0;
	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(i);
		CAPTURE(predicted[i]);
		worst = std::max(worst, std::abs((f64)result[i] - predicted[i]));
		CHECK(result[i] == doctest::Approx(predicted[i]).epsilon(0.04));
		CHECK(result[i] <= 1.001f);
	}
	MESSAGE("worst |furnace - F82-scaled prediction|: ", worst);
}

TEST_CASE("metallic lobe: furnace monotonic in base color") {
	// brighter metal reflects more energy, all else equal
	std::vector<f32> const grays = { 0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 1.0f };
	std::vector<f32> flat;
	for (f32 g : grays) {
		push_furnace_sample(flat, 0.5f, 0.4f, 0.0f, 1.0f, g, 1.0f, 1.0f);
	}
	auto const result = integrate_metal_furnace(flat, 200u, 400u);

	f32 prev = -1.0f;
	for (u32 i = 0; i < result.size(); ++i) {
		CAPTURE(grays[i]);
		CHECK(result[i] > prev);
		prev = result[i];
	}
}

// -----------------------------------------------------------------------------
// materialx independent cross-check (mirror configurations; see the
// shader comment for why the mu conventions coincide exactly there)
// -----------------------------------------------------------------------------

TEST_CASE("metallic lobe: matches materialx assembly on mirror configurations (grid)") {
	std::vector<f32v3> const colors = {
		skGoldColor, { 0.95f, 0.93f, 0.88f }, { 1.0f, 1.0f, 1.0f },
		{ 0.3f, 0.3f, 0.3f },
	};
	std::vector<f32> flat;
	for (f32v3 const & color : colors) {
		for (u32 m = 1; m <= 20; ++m) {
			for (u32 r = 1; r <= 10; ++r) {
				push_crosscheck_sample(
					flat, (f32)m / 20.0f, (f32)r / 10.0f,
					1.0f, color, 1.0f, 0.9f
				);
			}
		}
	}
	std::vector<f32v3> mine, ref;
	crosscheck_metal(flat, mine, ref);

	f64 maxRelDiff = 0.0;
	for (u32 i = 0; i < mine.size(); ++i) {
		CAPTURE(i);
		CHECK(mine[i].x == doctest::Approx(ref[i].x).epsilon(0.005));
		CHECK(mine[i].y == doctest::Approx(ref[i].y).epsilon(0.005));
		CHECK(mine[i].z == doctest::Approx(ref[i].z).epsilon(0.005));
		f64 const rel = (
			std::abs((f64)mine[i].x - (f64)ref[i].x)
			/ std::max((f64)ref[i].x, 1e-9)
		);
		maxRelDiff = std::max(maxRelDiff, rel);
	}
	MESSAGE("materialx mirror-grid crosscheck max relative diff: ", maxRelDiff);
}

TEST_CASE("metallic lobe: matches materialx assembly on mirror configurations (fuzz)") {
	std::vector<f32> flat;
	u32 seed = 4321u;
	auto next = [&]() -> f32 {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	};
	for (u32 i = 0; i < 2048u; ++i) {
		push_crosscheck_sample(
			flat, 0.02f + next() * 0.98f, 0.02f + next() * 0.98f,
			next(), { next(), next(), next() }, next(), next()
		);
	}
	std::vector<f32v3> mine, ref;
	crosscheck_metal(flat, mine, ref);

	for (u32 i = 0; i < mine.size(); ++i) {
		CAPTURE(i);
		CHECK(mine[i].x == doctest::Approx(ref[i].x).epsilon(0.005));
		CHECK(mine[i].y == doctest::Approx(ref[i].y).epsilon(0.005));
		CHECK(mine[i].z == doctest::Approx(ref[i].z).epsilon(0.005));
	}
}

// -----------------------------------------------------------------------------
// visual representation
// -----------------------------------------------------------------------------

namespace {

void write_metal_disk(
	f32 specularRoughness, f32 specularRoughnessAnisotropy, char const * path,
	f32 tonemapHeadroom
) {
	// conductor lobe * pi over the wo hemisphere projected to the tangent
	// disk, wi fixed at 60 degrees in +x, gold f0 -- the highlight core
	// carries the gold tint, the grazing rim whitens as F82 -> 1.
	// outside the disk renders as background black
	constexpr u32 kSize = 512;
	f32 const muI = 0.5f;
	f32v3 const wi = { std::sqrt(1.0f - muI * muI), 0.0f, muI };

	std::vector<f32> flat;
	std::vector<bool> valid(kSize * kSize, false);
	flat.reserve((size_t)kSize * kSize * 14);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const wy = 2.0f * (f32)y / (f32)(kSize - 1) - 1.0f;
		for (u32 x = 0; x < kSize; ++x) {
			f32 const wx = 2.0f * (f32)x / (f32)(kSize - 1) - 1.0f;
			f32 const r2 = wx * wx + wy * wy;
			bool const inside = r2 <= 1.0f;
			valid[y * kSize + x] = inside;
			f32v3 const wo = { wx, wy, inside ? std::sqrt(1.0f - r2) : 1.0f };
			push_metal_sample(
				flat, wi, wo, 1.0f, skGoldColor, 1.0f, 1.0f,
				specularRoughness, specularRoughnessAnisotropy
			);
		}
	}
	auto const result = evaluate_metal(flat);

	std::vector<f32> r(result.size(), 0.0f), g(result.size(), 0.0f), b(result.size(), 0.0f);
	for (u32 i = 0; i < result.size(); ++i) {
		if (!valid[i]) { continue; }
		r[i] = result[i].x * 3.14159265f / tonemapHeadroom;
		g[i] = result[i].y * 3.14159265f / tonemapHeadroom;
		b[i] = result[i].z * 3.14159265f / tonemapHeadroom;
	}

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(r, g, b, kSize, kSize, path, &nanCount);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);
}

} // namespace

TEST_CASE("metallic lobe: brdf disk, smooth gold") {
	write_metal_disk(
		0.03f, 0.0f, METALLIC_OUTPUT_DIR "metallic_disk_smooth.png", 8.0f
	);
}

TEST_CASE("metallic lobe: brdf disk, medium gold") {
	write_metal_disk(
		0.15f, 0.0f, METALLIC_OUTPUT_DIR "metallic_disk_medium.png", 3.0f
	);
}

TEST_CASE("metallic lobe: brdf disk, rough gold") {
	write_metal_disk(
		0.6f, 0.0f, METALLIC_OUTPUT_DIR "metallic_disk_rough.png", 2.0f
	);
}

TEST_CASE("metallic lobe: brdf disk, anisotropic gold") {
	write_metal_disk(
		0.12f, 0.85f, METALLIC_OUTPUT_DIR "metallic_disk_anisotropic.png", 4.0f
	);
}

TEST_CASE("metallic lobe: furnace heatmap (roughness x incidence, white metal)") {
	// the image spans the full parameter square (including the
	// unresolvable near-mirror band, shown honestly); the per-pixel
	// assertion only covers roughness >= 0.1, where 80x160 quadrature
	// actually resolves the lobe -- same domain reasoning as the strict
	// furnace test above
	constexpr u32 kSize = 256;
	std::vector<f32> flat;
	std::vector<f32> roughs(kSize);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const rough = 0.02f + 0.98f * (f32)y / (f32)(kSize - 1);
		roughs[y] = rough;
		for (u32 x = 0; x < kSize; ++x) {
			f32 const mu = 1.0f - 0.98f * (f32)x / (f32)(kSize - 1);
			push_furnace_sample(flat, mu, rough, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f);
		}
	}
	auto const result = integrate_metal_furnace(flat, 80u, 160u);

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		result, result, result, kSize, kSize,
		METALLIC_OUTPUT_DIR "metallic_furnace_roughness_incidence.png",
		&nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);

	f64 worst = 0.0;
	f64 worstResolvable = 0.0;
	for (u32 y = 0; y < kSize; ++y) {
		for (u32 x = 0; x < kSize; ++x) {
			f64 const d = std::abs((f64)result[y * kSize + x] - 1.0);
			worst = std::max(worst, d);
			if (roughs[y] >= 0.1f) {
				worstResolvable = std::max(worstResolvable, d);
				CAPTURE(x);
				CAPTURE(y);
				CHECK(result[y * kSize + x] == doctest::Approx(1.0).epsilon(0.09));
			}
		}
	}
	MESSAGE(
		"furnace heatmap worst |albedo - 1|: full ", worst,
		", resolvable domain (rough >= 0.1) ", worstResolvable
	);
}

TEST_CASE("metallic lobe: furnace heatmap (roughness x anisotropy, white metal)") {
	// image spans the full square; assertions cover roughness >= 0.1 and
	// anisotropy <= 0.8 -- past that alphaB collapses toward its 1e-5
	// floor and the needle-thin lobe is a pure quadrature casualty at
	// 80x160 steps (the strict 200x400 test above covers up to 0.95 with
	// a widened tolerance instead)
	constexpr u32 kSize = 256;
	std::vector<f32> flat;
	std::vector<f32> roughs(kSize), anisos(kSize);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const rough = 0.02f + 0.98f * (f32)y / (f32)(kSize - 1);
		roughs[y] = rough;
		for (u32 x = 0; x < kSize; ++x) {
			f32 const aniso = (f32)x / (f32)(kSize - 1);
			anisos[x] = aniso;
			push_furnace_sample(flat, 0.5f, rough, aniso, 1.0f, 1.0f, 1.0f, 1.0f);
		}
	}
	auto const result = integrate_metal_furnace(flat, 80u, 160u);

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		result, result, result, kSize, kSize,
		METALLIC_OUTPUT_DIR "metallic_furnace_roughness_anisotropy.png",
		&nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);

	f64 worst = 0.0;
	f64 worstResolvable = 0.0;
	for (u32 y = 0; y < kSize; ++y) {
		for (u32 x = 0; x < kSize; ++x) {
			f64 const d = std::abs((f64)result[y * kSize + x] - 1.0);
			worst = std::max(worst, d);
			if (roughs[y] >= 0.1f && anisos[x] <= 0.8f) {
				worstResolvable = std::max(worstResolvable, d);
				CAPTURE(x);
				CAPTURE(y);
				CHECK(result[y * kSize + x] == doctest::Approx(1.0).epsilon(0.09));
			}
		}
	}
	MESSAGE(
		"aniso furnace heatmap worst |albedo - 1|: full ", worst,
		", resolvable domain ", worstResolvable
	);
}

TEST_CASE("metallic lobe: furnace heatmap (incidence x base color)") {
	constexpr u32 kSize = 256;
	std::vector<f32> flat;
	for (u32 y = 0; y < kSize; ++y) {
		f32 const gray = (f32)y / (f32)(kSize - 1);
		for (u32 x = 0; x < kSize; ++x) {
			f32 const mu = 1.0f - 0.98f * (f32)x / (f32)(kSize - 1);
			push_furnace_sample(flat, mu, 0.4f, 0.0f, 1.0f, gray, 1.0f, 1.0f);
		}
	}
	auto const result = integrate_metal_furnace(flat, 80u, 160u);

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		result, result, result, kSize, kSize,
		METALLIC_OUTPUT_DIR "metallic_furnace_incidence_color.png",
		&nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);
	// 1.02 bound: the turquin fit overshoots true energy by up to ~0.8%
	// at bright f0 here; consistent with the dielectric furnace test's
	// own "generous" 1.05 bound for the same compensation family
	for (f32 v : result) {
		CHECK(v >= -1e-5f);
		CHECK(v <= 1.02f);
	}
}

} // TEST_SUITE("[headless]")
