#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// tests for openPbrSubsurfaceAlbedoToSigma
// (lib/ponder/shaders/util-material-openpbr-subsurface.glsl): the
// christensen & burley 2015 diffusion-reflectance inversion that turns
// openpbr's user-facing (subsurfaceColor, subsurfaceRadius *
// subsurfaceRadiusScale, subsurfaceScatterAnisotropy) into the
// (sigmaAbsorption, sigmaScattering) extinction pair a random-walk
// subsurface integrator needs.
//
// neither cull nor materialx has a real random-walk subsurface reference to
// crosscheck against: cull's own util-material-openpbr-subsurface.glsl is
// the same instantaneous-diffuse placeholder ponder had, and materialx's
// mx_subsurface_bsdf.glsl is a screen-space rasterizer approximation
// (fwidth-based curvature, punctual L/N) with no per-direction bsdf to diff
// against. in place of that crosscheck, this suite verifies the inversion
// against its own closed-form diffusion profile at f64 precision: solving
// for alpha' and then re-evaluating R_d(alpha') must reproduce the
// requested albedo, which is the analytic equivalent of a furnace test for
// this term.

namespace {

// -----------------------------------------------------------------------------
// f64 reference, transcribed from util-material-openpbr-subsurface.glsl
// -----------------------------------------------------------------------------

f64 ref_diffusion_reflectance(f64 const alphaReduced) {
	f64 const s = std::sqrt(3.0 * (1.0 - alphaReduced));
	return (
		0.5 * alphaReduced
		* (1.0 + std::exp(-4.0 / 3.0 * s))
		* std::exp(-s)
	);
}

f64 ref_invert_diffusion_reflectance(f64 const targetAlbedo) {
	f64 lo = 0.0;
	f64 hi = 1.0;
	for (int i = 0; i < 60; ++i) {
		f64 const mid = 0.5 * (lo + hi);
		if (ref_diffusion_reflectance(mid) < targetAlbedo) {
			lo = mid;
		} else {
			hi = mid;
		}
	}
	return 0.5 * (lo + hi);
}

constexpr f64 skRefMinSubsurfaceRadius = 1e-5;
constexpr f64 skRefMaxSubsurfaceAnisotropy = 0.95;

struct RefSigma { f64 sigmaAbsorption, sigmaScattering; };

RefSigma ref_albedo_to_sigma(
	f64 const albedo, f64 const radius, f64 const anisotropy
) {
	f64 const safeRadius = std::fmax(radius, skRefMinSubsurfaceRadius);
	f64 const g = std::fmin(anisotropy, skRefMaxSubsurfaceAnisotropy);
	f64 const alphaReduced = (
		ref_invert_diffusion_reflectance(std::clamp(albedo, 0.0, 1.0))
	);
	f64 const sigmaTReduced = 1.0 / safeRadius;
	f64 const sigmaSReduced = alphaReduced * sigmaTReduced;
	f64 const sigmaAbsorption = sigmaTReduced - sigmaSReduced;
	f64 const sigmaScattering = sigmaSReduced / (1.0 - g);
	return { sigmaAbsorption, sigmaScattering };
}

// -----------------------------------------------------------------------------
// subsurface_albedo_to_sigma.comp runner
// -----------------------------------------------------------------------------

struct SubsurfacePush {
	u64 inVa;
	u64 outVa;
	u32 pointCount;
};
static_assert(sizeof(SubsurfacePush) == 24);

struct SigmaPoint {
	f32v3 sigmaAbsorption;
	f32v3 sigmaScattering;
};

// in0 packs (albedo.rgb, anisotropy) and in1 packs (radius.rgb, unused),
// one pair per point -- must match subsurface_albedo_to_sigma.comp's layout
std::vector<SigmaPoint> run_subsurface_albedo_to_sigma(
	std::vector<f32v4> const & in0,
	std::vector<f32v4> const & in1
) {
	u32 const pointCount = static_cast<u32>(in0.size());
	REQUIRE(in1.size() == in0.size());

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "subsurface_albedo_to_sigma.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	std::vector<f32v4> interleaved(static_cast<u64>(pointCount) * 2u);
	for (u32 i = 0; i < pointCount; ++i) {
		interleaved[i * 2u + 0u] = in0[i];
		interleaved[i * 2u + 1u] = in1[i];
	}
	auto inBuf = test::upload_bytes(
		interleaved.data(), interleaved.size() * sizeof(f32v4)
	);
	auto outBuf = vkof::buffer_create({
		.byteCount = static_cast<u64>(pointCount) * 2u * sizeof(f32v4),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	SubsurfacePush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.pointCount = pointCount,
	};
	static constexpr u32 kLocalSize = 64;
	test::dispatch(pl, push, (pointCount + kLocalSize - 1u) / kLocalSize);

	auto raw = test::readback<f32v4>(outBuf, 0, pointCount * 2u);
	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);

	std::vector<SigmaPoint> result(pointCount);
	for (u32 i = 0; i < pointCount; ++i) {
		result[i].sigmaAbsorption = {
			raw[i * 2u + 0u].x, raw[i * 2u + 0u].y, raw[i * 2u + 0u].z
		};
		result[i].sigmaScattering = {
			raw[i * 2u + 1u].x, raw[i * 2u + 1u].y, raw[i * 2u + 1u].z
		};
	}
	return result;
}

} // namespace

TEST_SUITE("[headless]") {

// -----------------------------------------------------------------------------
// task: gpu f32 evaluation matches the f64 reference across a dense sweep,
// broadcasting one scalar albedo/radius to all three channels per point
// -----------------------------------------------------------------------------

TEST_CASE("subsurface albedo->sigma: gpu matches f64 reference, broadcast channels") {
	std::vector<f32v4> in0;
	std::vector<f32v4> in1;
	std::vector<RefSigma> refs;

	constexpr u32 kAlbedoSteps = 12u;
	constexpr u32 kRadiusSteps = 8u;
	constexpr u32 kAnisotropySteps = 5u;
	for (u32 a = 0; a < kAlbedoSteps; ++a) {
		f64 const albedo = (a + 0.5) / kAlbedoSteps;
		for (u32 r = 0; r < kRadiusSteps; ++r) {
			f64 const radius = 0.02 + 2.0 * (r + 0.5) / kRadiusSteps;
			for (u32 g = 0; g < kAnisotropySteps; ++g) {
				f64 const anisotropy = -0.9 + 1.8 * g / (kAnisotropySteps - 1u);
				in0.push_back({
					(f32)albedo, (f32)albedo, (f32)albedo, (f32)anisotropy
				});
				in1.push_back({ (f32)radius, (f32)radius, (f32)radius, 0.0f });
				refs.push_back(ref_albedo_to_sigma(albedo, radius, anisotropy));
			}
		}
	}

	auto const gpu = run_subsurface_albedo_to_sigma(in0, in1);
	REQUIRE(gpu.size() == refs.size());

	f64 worstAbsRelErr = 0.0;
	f64 worstScatRelErr = 0.0;
	for (u32 i = 0; i < gpu.size(); ++i) {
		f64 const gpuA = gpu[i].sigmaAbsorption.x;
		f64 const gpuS = gpu[i].sigmaScattering.x;
		// combined abs+rel tolerance: a pure relative check blows up near
		// the albedo->0/1 extremes where one of sigmaAbsorption/
		// sigmaScattering itself approaches zero and f32-vs-f64 bisection
		// rounding dominates a tiny denominator
		f64 const relA = (
			std::fabs(gpuA - refs[i].sigmaAbsorption)
			/ (std::fabs(refs[i].sigmaAbsorption) + 1e-3)
		);
		f64 const relS = (
			std::fabs(gpuS - refs[i].sigmaScattering)
			/ (std::fabs(refs[i].sigmaScattering) + 1e-3)
		);
		worstAbsRelErr = std::fmax(worstAbsRelErr, relA);
		worstScatRelErr = std::fmax(worstScatRelErr, relS);
		// every channel of a broadcast input must agree exactly with itself
		CHECK(gpu[i].sigmaAbsorption.x == gpu[i].sigmaAbsorption.y);
		CHECK(gpu[i].sigmaAbsorption.x == gpu[i].sigmaAbsorption.z);
		CHECK(gpu[i].sigmaScattering.x == gpu[i].sigmaScattering.y);
		CHECK(gpu[i].sigmaScattering.x == gpu[i].sigmaScattering.z);
	}
	CAPTURE(worstAbsRelErr);
	CAPTURE(worstScatRelErr);
	CHECK(worstAbsRelErr < 1e-3);
	CHECK(worstScatRelErr < 1e-3);
}

TEST_CASE("subsurface albedo->sigma: per-channel values are independent") {
	std::vector<f32v4> const in0 { { 0.1f, 0.5f, 0.9f, 0.3f } };
	std::vector<f32v4> const in1 { { 0.1f, 0.5f, 1.5f, 0.0f } };
	auto const gpu = run_subsurface_albedo_to_sigma(in0, in1);
	REQUIRE(gpu.size() == 1);

	f64 const refR = ref_albedo_to_sigma(0.1, 0.1, 0.3).sigmaAbsorption;
	f64 const refG = ref_albedo_to_sigma(0.5, 0.5, 0.3).sigmaAbsorption;
	f64 const refB = ref_albedo_to_sigma(0.9, 1.5, 0.3).sigmaAbsorption;
	CHECK(gpu[0].sigmaAbsorption.x == doctest::Approx(refR).epsilon(1e-3));
	CHECK(gpu[0].sigmaAbsorption.y == doctest::Approx(refG).epsilon(1e-3));
	CHECK(gpu[0].sigmaAbsorption.z == doctest::Approx(refB).epsilon(1e-3));
	// channels must not collapse to a shared value just because they ran
	// through the same vectorized bisection lanes
	CHECK(
		std::fabs(gpu[0].sigmaAbsorption.x - gpu[0].sigmaAbsorption.z) > 0.01f
	);
	CHECK(
		std::fabs(gpu[0].sigmaScattering.x - gpu[0].sigmaScattering.z) > 0.01f
	);
}

// -----------------------------------------------------------------------------
// task: degenerate/bounds cases stay finite and match the analytic limits
// -----------------------------------------------------------------------------

TEST_CASE("subsurface albedo->sigma: degenerate and bounds cases") {
	std::vector<f32v4> const in0 {
		{ 0.0f, 0.0f, 0.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 0.5f, 0.5f, 0.5f, 0.0f },
		{ 0.5f, 0.5f, 0.5f, 1.0f },
		{ 0.5f, 0.5f, 0.5f, -1.0f },
		{ -0.3f, 1.4f, 0.5f, 0.0f },
	};
	std::vector<f32v4> const in1 {
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
	};
	auto const gpu = run_subsurface_albedo_to_sigma(in0, in1);
	REQUIRE(gpu.size() == in0.size());

	for (auto const & p : gpu) {
		CHECK(std::isfinite(p.sigmaAbsorption.x));
		CHECK(std::isfinite(p.sigmaAbsorption.y));
		CHECK(std::isfinite(p.sigmaAbsorption.z));
		CHECK(std::isfinite(p.sigmaScattering.x));
		CHECK(std::isfinite(p.sigmaScattering.y));
		CHECK(std::isfinite(p.sigmaScattering.z));
		CHECK(p.sigmaAbsorption.x >= 0.0f);
		CHECK(p.sigmaScattering.x >= 0.0f);
	}

	// albedo 0: pure absorption, no scattering
	CHECK(gpu[0].sigmaScattering.x == doctest::Approx(0.0f).epsilon(1e-4));
	CHECK(gpu[0].sigmaAbsorption.x == doctest::Approx(1.0f).epsilon(1e-3));

	// albedo 1: pure scattering, no absorption
	CHECK(gpu[1].sigmaAbsorption.x == doctest::Approx(0.0f).epsilon(1e-4));
	CHECK(gpu[1].sigmaScattering.x > 0.0f);

	// radius 0 floors to skMinSubsurfaceRadius rather than exploding
	CHECK(gpu[2].sigmaAbsorption.x > 1e3f);

	// anisotropy 1.0 clamps to skMaxSubsurfaceAnisotropy (0.95) instead of
	// dividing by (1 - g) = 0 outright
	f64 const refClampedG = (
		ref_albedo_to_sigma(0.5, 1.0, 1.0).sigmaScattering
	);
	CHECK(
		gpu[3].sigmaScattering.x
		== doctest::Approx((f32)refClampedG).epsilon(1e-3)
	);

	// backward anisotropy (g=-1) does not need clamping or guarding
	f64 const refBackwardG = (
		ref_albedo_to_sigma(0.5, 1.0, -1.0).sigmaScattering
	);
	CHECK(
		gpu[4].sigmaScattering.x
		== doctest::Approx((f32)refBackwardG).epsilon(1e-3)
	);

	// out-of-[0,1] albedo clamps rather than producing a bisection that
	// never converges into range: x clamps -0.3 -> 0 (pure absorption,
	// same as the exact-zero case above), z is untouched (0.5, in-range)
	CHECK(gpu[5].sigmaAbsorption.x == doctest::Approx(1.0f).epsilon(1e-3));
	CHECK(gpu[5].sigmaScattering.x == doctest::Approx(0.0f).epsilon(1e-4));
	CHECK(gpu[5].sigmaAbsorption.z >= 0.0f);
}

// -----------------------------------------------------------------------------
// task: energy check -- inverting the returned sigma pair back to alpha'
// and re-evaluating R_d(alpha') must reproduce the requested albedo. this
// stands in for the materialx crosscheck every other openpbr term has: see
// the file header for why a real one isn't available for this term.
// -----------------------------------------------------------------------------

TEST_CASE("subsurface albedo->sigma: round-trip reproduces requested albedo") {
	std::vector<f32v4> in0;
	std::vector<f32v4> in1;
	std::vector<f64> requestedAlbedo;
	std::vector<f64> anisotropies;

	constexpr u32 kAlbedoSteps = 10u;
	constexpr u32 kRadiusSteps = 6u;
	constexpr u32 kAnisotropySteps = 4u;
	for (u32 a = 0; a < kAlbedoSteps; ++a) {
		// clear the alpha'=0/1 endpoints: bisection lands exactly on the
		// bracket boundary there, and dividing back out is a 0/0 in the
		// round-trip check below, not a defect in the forward inversion
		f64 const albedo = 0.02 + 0.96 * (a + 0.5) / kAlbedoSteps;
		for (u32 r = 0; r < kRadiusSteps; ++r) {
			f64 const radius = 0.05 + 2.0 * (r + 0.5) / kRadiusSteps;
			for (u32 g = 0; g < kAnisotropySteps; ++g) {
				f64 const anisotropy = -0.8 + 1.6 * g / (kAnisotropySteps - 1u);
				in0.push_back({
					(f32)albedo, (f32)albedo, (f32)albedo, (f32)anisotropy
				});
				in1.push_back({ (f32)radius, (f32)radius, (f32)radius, 0.0f });
				requestedAlbedo.push_back(albedo);
				anisotropies.push_back(anisotropy);
			}
		}
	}

	auto const gpu = run_subsurface_albedo_to_sigma(in0, in1);
	REQUIRE(gpu.size() == requestedAlbedo.size());

	f64 worstErr = 0.0;
	for (u32 i = 0; i < gpu.size(); ++i) {
		f64 const g = std::fmin(anisotropies[i], skRefMaxSubsurfaceAnisotropy);
		f64 const sigmaSReduced = (f64)gpu[i].sigmaScattering.x * (1.0 - g);
		f64 const sigmaTReduced = sigmaSReduced + (f64)gpu[i].sigmaAbsorption.x;
		f64 const alphaReduced = sigmaSReduced / sigmaTReduced;
		f64 const recoveredAlbedo = ref_diffusion_reflectance(alphaReduced);
		f64 const err = std::fabs(recoveredAlbedo - requestedAlbedo[i]);
		worstErr = std::fmax(worstErr, err);
	}
	CAPTURE(worstErr);
	CHECK(worstErr < 5e-3);
}

// -----------------------------------------------------------------------------
// task: visual sweep over (albedo, radius) at a couple of fixed anisotropy
// values; non-finite values render as pure red rather than failing silently
// -----------------------------------------------------------------------------

void subsurface_heatmap(f32 const anisotropy, char const * const path) {
	constexpr u32 kWidth = 96u;
	constexpr u32 kHeight = 96u;
	std::vector<f32v4> in0(kWidth * kHeight);
	std::vector<f32v4> in1(kWidth * kHeight);
	for (u32 y = 0; y < kHeight; ++y) {
		f32 const radius = 0.05f + 2.0f * (y + 0.5f) / kHeight;
		for (u32 x = 0; x < kWidth; ++x) {
			f32 const albedo = (x + 0.5f) / kWidth;
			u32 const i = y * kWidth + x;
			in0[i] = { albedo, albedo, albedo, anisotropy };
			in1[i] = { radius, radius, radius, 0.0f };
		}
	}
	auto const gpu = run_subsurface_albedo_to_sigma(in0, in1);

	std::vector<f32> r(kWidth * kHeight);
	std::vector<f32> g(kWidth * kHeight, 0.0f);
	std::vector<f32> b(kWidth * kHeight);
	for (u32 i = 0; i < gpu.size(); ++i) {
		// bounded x/(1+x) mapping so any finite non-negative sigma lands
		// in [0, 1); a non-finite sigma maps to inf/inf = nan and still
		// trips write_heatmap_png's red fallback
		f32 const sa = gpu[i].sigmaAbsorption.x;
		f32 const ss = gpu[i].sigmaScattering.x;
		r[i] = sa / (1.0f + sa);
		b[i] = ss / (1.0f + ss);
	}

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		r, g, b, kWidth, kHeight, path, &nanCount
	);
	CHECK(ok);
	CHECK(nanCount == 0);
}

TEST_CASE("subsurface albedo->sigma: heatmap, isotropic") {
	subsurface_heatmap(
		0.0f,
		(std::string(SUBSURFACE_OUTPUT_DIR) + "subsurface_albedo_sigma_isotropic.png").c_str()
	);
}

TEST_CASE("subsurface albedo->sigma: heatmap, forward-scattering") {
	subsurface_heatmap(
		0.8f,
		(std::string(SUBSURFACE_OUTPUT_DIR) + "subsurface_albedo_sigma_forward.png").c_str()
	);
}

} // TEST_SUITE("[headless]")
