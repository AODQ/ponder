#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <mor/mor.hpp>
#include <ponder/energy-tables.hpp>
#include <ponder/zeltner-tables.hpp>
#include "util.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Where the thin-walled subsurface sheet ends up once it is *assembled* into
// the full material.
//
// The suite already had the two ends of this story and not the middle:
//
//   test-thin-walled-subsurface.cpp  -- the sheet lobe in isolation (shape,
//     two-sidedness, E_R + E_T == subsurface_color, pdf, chi-square, the
//     g = +-1 hemisphere collapse). Calls openPbrThinWalledSubsurfaceEvaluateF
//     directly.
//   test-diffuse-transmission.cpp    -- the gltf loader mapping
//     (diffuseTransmissionFactor -> subsurfaceWeight, colour -> subsurfaceColor,
//     g = 1, thinWalled = 1, the A-channel swizzle).
//
// Neither one runs openPbrEvaluateF. So nothing checked the step in between:
// subsurfaceWeight blending the sheet against the base diffuse lobe inside
// fTDielectricBase, underneath the specular interface, coat and fuzz. The
// glTF contract
//
//   reflected   = (1 - factor) * baseColor
//   transmitted = factor * diffuseTransmissionColor
//
// is a statement about that assembled material. It is what decides whether a
// peel with factor 1 goes black when front-lit and only lights up from
// behind, or whether it stays bright and washed out from every direction --
// which is the visible question this file exists to answer.
//
// The assertions are written as ratios rather than closed forms (reflected(w)
// == (1 - w) * reflected(0), transmitted(w) == w * transmitted(1)) so they
// pin the split exactly without depending on the diffuse lobe's own
// normalization, which the isolation tests already own.
//
// What it found, 2026-08-04:
//
//   - The split itself is exact. factor and anisotropy both move the
//     reflected/transmitted shares to their authored values to 1e-4, and the
//     uncoated material emits exactly (1 - w) * baseColor + w * subsurfaceColor.
//     A peel authored at factor 1 really does have zero diffuse reflection.
//   - But a *coat* or a *sheen* over a thin-walled diffuse transmitter gains
//     energy at grazing. openPbrEvaluateF returns early on the transmitted
//     branch, so the sheet is attenuated by the specular interface's essThin
//     and by nothing else -- the coat and fuzz are skipped there, while still
//     adding their own reflection on the other side. Light therefore reflects
//     off the coat *and* transmits through the sheet at full strength.
//     Converged directional albedo at 83 degrees: 1.096 with a coat, 1.086
//     with sheen, 1.215 with both, against 0.919 for the uncoated sheet.
//     Harmless near normal incidence (~4% Fresnel), 20%+ at a silhouette.
// ---------------------------------------------------------------------------

namespace {

u32 shared_zeltner_handle() {
	static ponder::ZeltnerTables const tables = ponder::zeltner_tables_create();
	return tables.zeltnerLtcParamHandle;
}

u32 shared_kulla_conty_handle() {
	static ponder::EnergyTables const tables = ponder::energy_tables_create();
	return tables.kullaContyEnergyHandle;
}

// layout must match diffuse_transmission_placement.comp's scalar push block
struct PlacementPush {
	u64 outVa;
	u32 kullaContyEnergyHandle;
	u32 zeltnerLtcParamHandle;
	u32 mode;
	u32 sampleCount;
	f32 wiTheta;
	f32 baseColor[3];
	f32 subsurfaceColor[3];
	f32 subsurfaceWeight;
	f32 subsurfaceScatterAnisotropy;
	f32 geometryThinWalled;
	f32 specularIor;
	f32 specularRoughness;
	f32 baseDiffuseRoughness;
	f32 coatWeight;
	f32 coatRoughness;
	f32 fuzzWeight;
	f32 fuzzRoughness;
	u32 subTheta;
	u32 subPhi;
};

struct Config {
	f32 baseColor[3] { 0.7f, 0.3f, 0.1f };
	f32 subsurfaceColor[3] { 0.9f, 0.8f, 0.6f };
	f32 subsurfaceWeight { 0.0f };
	// the value mor's gltf diffuse-transmission path always writes: the whole
	// sheet on the transmitted side
	f32 subsurfaceScatterAnisotropy { 1.0f };
	f32 geometryThinWalled { 1.0f };
	// ior 1 removes the dielectric interface entirely (f0 = 0, so no specular
	// lobe, no (1 - E_spec) factor on the base, and essThin = 0 on the
	// transmitted branch), which isolates the diffuse/sheet split. cases that
	// want the real layered material set this to 1.5
	f32 specularIor { 1.0f };
	f32 specularRoughness { 0.3f };
	f32 baseDiffuseRoughness { 0.0f };
	f32 coatWeight { 0.0f };
	f32 coatRoughness { 0.1f };
	f32 fuzzWeight { 0.0f };
	f32 fuzzRoughness { 0.5f };
};

struct Rgb {
	f64 r;
	f64 g;
	f64 b;
};

constexpr u32 kLocalSize = 64u;

std::vector<f32> run_placement(
	Config const & config,
	u32 const mode,
	u32 const sampleCount,
	f32 const wiTheta,
	u32 const subTheta = 16u,
	u32 const subPhi = 64u
) {
	char const * const includePaths[] = { PONDER_SHADER_DIR, TEST_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "slang/entry/diffuse_transmission_placement.slang",
		.includePaths = srat::slice<char const * const>(includePaths, 2),
	});
	REQUIRE(pl.id != 0);

	u32 const floatCount = sampleCount * 3u;
	auto buf = vkof::buffer_create({
		.byteCount = floatCount * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	PlacementPush const push {
		.outVa = vkof::buffer_virtual_address(buf),
		.kullaContyEnergyHandle = shared_kulla_conty_handle(),
		.zeltnerLtcParamHandle = shared_zeltner_handle(),
		.mode = mode,
		.sampleCount = sampleCount,
		.wiTheta = wiTheta,
		.baseColor = {
			config.baseColor[0], config.baseColor[1], config.baseColor[2],
		},
		.subsurfaceColor = {
			config.subsurfaceColor[0],
			config.subsurfaceColor[1],
			config.subsurfaceColor[2],
		},
		.subsurfaceWeight = config.subsurfaceWeight,
		.subsurfaceScatterAnisotropy = config.subsurfaceScatterAnisotropy,
		.geometryThinWalled = config.geometryThinWalled,
		.specularIor = config.specularIor,
		.specularRoughness = config.specularRoughness,
		.baseDiffuseRoughness = config.baseDiffuseRoughness,
		.coatWeight = config.coatWeight,
		.coatRoughness = config.coatRoughness,
		.fuzzWeight = config.fuzzWeight,
		.fuzzRoughness = config.fuzzRoughness,
		.subTheta = subTheta,
		.subPhi = subPhi,
	};
	test::dispatch(pl, push, (sampleCount + kLocalSize - 1u) / kLocalSize);

	auto out = test::readback<f32>(buf, 0, floatCount);
	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
	return out;
}

constexpr u32 kSweepCount = 256u;
constexpr u32 kEnergyBands = 128u;

// sweep index -> outgoing theta, matching the shader
f64 sweep_theta(u32 const index) {
	f64 const pi = 3.14159265358979323846;
	return pi * ((f64)index + 0.5) / (f64)kSweepCount;
}

// mean of the reflected (theta < pi/2) or transmitted (theta > pi/2) half of
// a sweep. the two halves are the two sides of the sheet
Rgb sweep_half_mean(std::vector<f32> const & sweep, bool const reflected) {
	f64 const halfPi = 3.14159265358979323846 * 0.5;
	Rgb total { 0.0, 0.0, 0.0 };
	u32 count = 0u;
	for (u32 i = 0u; i < kSweepCount; ++i) {
		bool const isReflected = sweep_theta(i) < halfPi;
		if (isReflected != reflected) {
			continue;
		}
		total.r += (f64)sweep[i * 3u + 0u];
		total.g += (f64)sweep[i * 3u + 1u];
		total.b += (f64)sweep[i * 3u + 2u];
		++count;
	}
	REQUIRE(count > 0u);
	return { total.r / count, total.g / count, total.b / count };
}

// directional albedo split into the two hemispheres
struct Energy {
	Rgb reflected;
	Rgb transmitted;
};

Energy energy_split(std::vector<f32> const & bands) {
	Energy energy { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } };
	for (u32 i = 0u; i < kEnergyBands; ++i) {
		Rgb & target = (
			i < kEnergyBands / 2u ? energy.reflected : energy.transmitted
		);
		target.r += (f64)bands[i * 3u + 0u];
		target.g += (f64)bands[i * 3u + 1u];
		target.b += (f64)bands[i * 3u + 2u];
	}
	return energy;
}

f64 max_component(Rgb const & c) {
	return std::fmax(c.r, std::fmax(c.g, c.b));
}

// grazing incidence, where the transmitted branch's (1 - essThin) gate bites
constexpr f32 skGrazingWiTheta = 1.45f;
constexpr f32 skNormalWiTheta = 0.3f;

} // namespace

TEST_SUITE("[headless]") {

// ---------------------------------------------------------------------------
// -- the gltf split, at the assembled-material level
// ---------------------------------------------------------------------------

TEST_CASE("diffuse transmission placement: factor splits reflect vs transmit") {
	// with the dielectric interface removed, openPbrEvaluateF's whole output
	// is the diffuse/sheet mix, so the split is directly readable
	Config config;
	config.subsurfaceWeight = 0.0f;
	auto const atZero = run_placement(config, 0u, kSweepCount, skNormalWiTheta);
	config.subsurfaceWeight = 1.0f;
	auto const atOne = run_placement(config, 0u, kSweepCount, skNormalWiTheta);

	Rgb const reflectedAtZero = sweep_half_mean(atZero, /*reflected=*/true);
	Rgb const transmittedAtOne = sweep_half_mean(atOne, /*reflected=*/false);
	REQUIRE(max_component(reflectedAtZero) > 0.0);
	REQUIRE(max_component(transmittedAtOne) > 0.0);

	// factor 0 must be a plain opaque diffuse surface: nothing transmitted
	Rgb const transmittedAtZero = sweep_half_mean(atZero, /*reflected=*/false);
	CHECK(max_component(transmittedAtZero) == doctest::Approx(0.0).epsilon(1e-6));

	// factor 1 must be a pure transmitter: no diffuse reflection at all.
	// this is the one that decides whether a peel authored at factor 1 goes
	// dark when front-lit
	Rgb const reflectedAtOne = sweep_half_mean(atOne, /*reflected=*/true);
	CAPTURE(reflectedAtOne.r);
	CAPTURE(reflectedAtOne.g);
	CAPTURE(reflectedAtOne.b);
	CHECK(max_component(reflectedAtOne) == doctest::Approx(0.0).epsilon(1e-6));

	std::printf("diffuse transmission placement -- factor sweep\n");
	for (f32 const weight : { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f }) {
		config.subsurfaceWeight = weight;
		auto const sweep = run_placement(config, 0u, kSweepCount, skNormalWiTheta);
		Rgb const reflected = sweep_half_mean(sweep, /*reflected=*/true);
		Rgb const transmitted = sweep_half_mean(sweep, /*reflected=*/false);
		f64 const reflectedFraction = reflected.r / reflectedAtZero.r;
		f64 const transmittedFraction = transmitted.r / transmittedAtOne.r;
		std::printf(
			"  factor %.2f | reflected %.4f (want %.4f) | transmitted %.4f"
			" (want %.4f)\n",
			(f64)weight, reflectedFraction, 1.0 - (f64)weight,
			transmittedFraction, (f64)weight
		);
		CAPTURE(weight);
		CHECK(reflectedFraction == doctest::Approx(1.0 - (f64)weight).epsilon(1e-4));
		CHECK(transmittedFraction == doctest::Approx((f64)weight).epsilon(1e-4));
	}
}

TEST_CASE("diffuse transmission placement: each side carries its own colour") {
	// reflected light must be tinted by baseColor and transmitted light by
	// subsurfaceColor -- swapping them, or letting one leak into the other,
	// is exactly what turns an orange peel whitish
	Config config;
	config.baseColor[0] = 0.8f;
	config.baseColor[1] = 0.2f;
	config.baseColor[2] = 0.05f;
	config.subsurfaceColor[0] = 0.3f;
	config.subsurfaceColor[1] = 0.6f;
	config.subsurfaceColor[2] = 0.9f;
	config.subsurfaceWeight = 0.5f;
	auto const sweep = run_placement(config, 0u, kSweepCount, skNormalWiTheta);

	Rgb const reflected = sweep_half_mean(sweep, /*reflected=*/true);
	Rgb const transmitted = sweep_half_mean(sweep, /*reflected=*/false);
	REQUIRE(reflected.r > 0.0);
	REQUIRE(transmitted.b > 0.0);

	// hue, normalized away from the shared lobe magnitude
	CHECK(
		reflected.g / reflected.r
		== doctest::Approx((f64)config.baseColor[1] / config.baseColor[0])
			.epsilon(1e-4)
	);
	CHECK(
		reflected.b / reflected.r
		== doctest::Approx((f64)config.baseColor[2] / config.baseColor[0])
			.epsilon(1e-4)
	);
	CHECK(
		transmitted.g / transmitted.r
		== doctest::Approx(
			(f64)config.subsurfaceColor[1] / config.subsurfaceColor[0]
		).epsilon(1e-4)
	);
	CHECK(
		transmitted.b / transmitted.r
		== doctest::Approx(
			(f64)config.subsurfaceColor[2] / config.subsurfaceColor[0]
		).epsilon(1e-4)
	);
}

TEST_CASE("diffuse transmission placement: anisotropy moves the split") {
	// g is the knob mor sets to 1 for gltf; the sheet's own tests pin
	// 1/2 (1 -+ g) in isolation, this checks the assembled material still
	// obeys it once subsurfaceWeight has blended it against the base
	Config config;
	config.subsurfaceWeight = 1.0f;
	std::printf("diffuse transmission placement -- anisotropy sweep\n");
	for (f32 const anisotropy : { -1.0f, -0.5f, 0.0f, 0.5f, 1.0f }) {
		config.subsurfaceScatterAnisotropy = anisotropy;
		auto const bands = run_placement(config, 1u, kEnergyBands, skNormalWiTheta);
		Energy const energy = energy_split(bands);
		f64 const total = energy.reflected.r + energy.transmitted.r;
		REQUIRE(total > 0.0);
		f64 const transmittedShare = energy.transmitted.r / total;
		f64 const want = 0.5 * (1.0 + (f64)anisotropy);
		std::printf(
			"  g %+.2f | transmitted share %.4f (want %.4f)\n",
			(f64)anisotropy, transmittedShare, want
		);
		CAPTURE(anisotropy);
		CHECK(transmittedShare == doctest::Approx(want).epsilon(0.01));
	}
}

TEST_CASE("diffuse transmission placement: assembled energy matches authoring") {
	// the "over tuned" check. with no dielectric interface the material may
	// emit exactly what it was authored to emit and no more:
	//   E_reflected + E_transmitted == (1 - w) * baseColor + w * subsurfaceColor
	Config config;
	std::printf("diffuse transmission placement -- energy vs authoring\n");
	for (f32 const weight : { 0.0f, 0.5f, 1.0f }) {
		config.subsurfaceWeight = weight;
		auto const bands = run_placement(config, 1u, kEnergyBands, skNormalWiTheta);
		Energy const energy = energy_split(bands);
		f64 const totalR = energy.reflected.r + energy.transmitted.r;
		f64 const wantR = (
			(1.0 - (f64)weight) * config.baseColor[0]
			+ (f64)weight * config.subsurfaceColor[0]
		);
		std::printf(
			"  factor %.2f | E_total %.4f (authored %.4f) | E_r %.4f E_t %.4f\n",
			(f64)weight, totalR, wantR, energy.reflected.r, energy.transmitted.r
		);
		CAPTURE(weight);
		CHECK(totalR == doctest::Approx(wantR).epsilon(0.01));
		// nothing may emit more than it receives
		CHECK(totalR < 1.0);
	}
}

TEST_CASE("diffuse transmission placement: layered material stays bounded") {
	// the real thing: dielectric interface at ior 1.5, plus a coat and fuzz.
	// no closed form here, but the layered material must still not gain
	// energy, and the transmitted branch must still respond to the factor
	Config config;
	config.specularIor = 1.5f;
	config.coatWeight = 1.0f;
	config.fuzzWeight = 0.3f;

	std::printf("diffuse transmission placement -- layered\n");
	for (f32 const wiTheta : { skNormalWiTheta, skGrazingWiTheta }) {
		config.subsurfaceWeight = 1.0f;
		// the coat lobe is sharp enough that the default grid under-reports;
		// the attribution case below shows this resolution is converged
		auto const bands = (
			run_placement(config, 1u, kEnergyBands, wiTheta, 32u, 256u)
		);
		Energy const energy = energy_split(bands);
		f64 const total = max_component(energy.reflected)
			+ max_component(energy.transmitted);
		std::printf(
			"  wiTheta %.2f | E_r %.4f | E_t %.4f | total %.4f\n",
			(f64)wiTheta, max_component(energy.reflected),
			max_component(energy.transmitted), total
		);
		CAPTURE(wiTheta);
		CHECK(std::isfinite(total));
		CHECK(total < 1.0);
	}

	// the transmitted branch is gated by (1 - essThin), the dielectric
	// interface's directional albedo, which rises toward grazing. so a
	// layered sheet must transmit strictly less at grazing than head-on --
	// this is what makes a silhouette rim specular rather than subsurface
	config.subsurfaceWeight = 1.0f;
	auto const headOn = (
		run_placement(config, 1u, kEnergyBands, skNormalWiTheta, 32u, 256u)
	);
	auto const grazing = (
		run_placement(config, 1u, kEnergyBands, skGrazingWiTheta, 32u, 256u)
	);
	f64 const headOnTransmitted = max_component(energy_split(headOn).transmitted);
	f64 const grazingTransmitted = max_component(energy_split(grazing).transmitted);
	CAPTURE(headOnTransmitted);
	CAPTURE(grazingTransmitted);
	CHECK(grazingTransmitted < headOnTransmitted);
}

TEST_CASE("diffuse transmission placement: uncoated sheet is energy bounded") {
	// the MandarinOrange configuration: a thin diffuse-transmissive sheet
	// under the plain dielectric interface, with no coat and no sheen. this
	// must stay bounded at every incidence, and it does -- which is what
	// makes the whitish peel an authoring result rather than a renderer one.
	// the coat/fuzz case above is a different configuration and does gain
	Config config;
	config.specularIor = 1.5f;
	config.specularRoughness = 0.39f;
	config.coatWeight = 0.0f;
	config.fuzzWeight = 0.0f;
	config.subsurfaceWeight = 1.0f;
	std::printf("diffuse transmission placement -- uncoated (mandarin config)\n");
	for (f32 const wiTheta : { 0.3f, 1.0f, 1.3f, 1.45f, 1.53f }) {
		auto const bands = (
			run_placement(config, 1u, kEnergyBands, wiTheta, 32u, 256u)
		);
		Energy const energy = energy_split(bands);
		f64 const total = energy.reflected.r + energy.transmitted.r;
		std::printf("  wiTheta %.2f | E_total %.4f\n", (f64)wiTheta, total);
		CAPTURE(wiTheta);
		CHECK(std::isfinite(total));
		CHECK(total < 1.0);
	}
}

TEST_CASE("diffuse transmission placement: which layer gains energy") {
	// diagnostic for the >1 albedo the layered case reports at grazing:
	// which layer is responsible, and is the number converged
	struct LayerCase {
		char const * name;
		f32 specularIor;
		f32 coatWeight;
		f32 fuzzWeight;
	};
	LayerCase const cases[] = {
		{ "sheet only (ior 1)", 1.0f, 0.0f, 0.0f },
		{ "+ dielectric 1.5", 1.5f, 0.0f, 0.0f },
		{ "+ coat", 1.5f, 1.0f, 0.0f },
		{ "+ fuzz", 1.5f, 0.0f, 0.3f },
		{ "+ coat + fuzz", 1.5f, 1.0f, 0.3f },
	};
	std::printf("diffuse transmission placement -- layer attribution (E_total, r)\n");
	for (LayerCase const & layerCase : cases) {
		Config config;
		config.subsurfaceWeight = 1.0f;
		config.specularIor = layerCase.specularIor;
		config.coatWeight = layerCase.coatWeight;
		config.fuzzWeight = layerCase.fuzzWeight;
		std::printf("  %-20s", layerCase.name);
		for (f32 const wiTheta : { 0.3f, 1.0f, 1.3f, 1.45f, 1.53f }) {
			auto const bands = run_placement(
				config, 1u, kEnergyBands, wiTheta, 32u, 256u
			);
			Energy const energy = energy_split(bands);
			std::printf(" %.3f", energy.reflected.r + energy.transmitted.r);
		}
		std::printf("\n");
	}
	std::printf("    (wiTheta = 0.30 1.00 1.30 1.45 1.53)\n");

	// convergence: the grazing number must not move with resolution
	Config config;
	config.subsurfaceWeight = 1.0f;
	config.specularIor = 1.5f;
	config.coatWeight = 1.0f;
	config.fuzzWeight = 0.3f;
	std::printf("  convergence at wiTheta 1.45:");
	for (u32 const sub : { 8u, 16u, 32u, 64u }) {
		auto const bands = run_placement(
			config, 1u, kEnergyBands, 1.45f, sub, sub * 8u
		);
		Energy const energy = energy_split(bands);
		std::printf(
			" %ux%u=%.4f", sub, sub * 8u,
			energy.reflected.r + energy.transmitted.r
		);
	}
	std::printf("\n");
}

// ---------------------------------------------------------------------------
// -- the asset the question came from
// ---------------------------------------------------------------------------

TEST_CASE("diffuse transmission placement: MandarinOrange authored values") {
	// this asset is the reason the file exists. it packs strength in the A
	// channel and colour in the RGB of one shared png, and authors
	// diffuseTransmissionFactor = 1, so wherever that alpha reaches 1 the
	// material above becomes a pure transmitter with no diffuse reflection.
	// pinning the loaded values here keeps "the peel is whitish" answerable
	// as an authoring question rather than a rendering one
	std::string const modelPath = (
		std::string(REPO_DIR)
		+ "/assets/models-categorized/test-transparency/MandarinOrange"
		+ "/MandarinOrange.gltf"
	);
	if (!std::filesystem::exists(modelPath)) {
		MESSAGE("MandarinOrange asset not present, skipping");
		return;
	}

	mor::Scene scene = mor::scene_create();
	mor::scene_load_gltf(scene, modelPath.c_str());
	REQUIRE(mor::scene_material_count(scene) > 0u);

	i32 index = -1;
	for (u32 i = 0u; i < mor::scene_material_count(scene); ++i) {
		if (mor::scene_material_name(scene, i) == std::string("MandarinOrange")) {
			index = (i32)i;
			break;
		}
	}
	REQUIRE(index >= 0);
	GpuMorMaterial const m = mor::scene_material_get(scene, (u32)index);

	// factor 1, with the per-texel strength coming from the texture's alpha
	CHECK(m.subsurfaceWeight.r == doctest::Approx(1.0f));
	CHECK(m.subsurfaceWeight.texture != 0u);
	CHECK(
		(m.subsurfaceWeight.swizzle & MOR_MATERIAL_SWIZZLE_CHANNEL_MASK) == 3
	);
	// colour from the same image's rgb, which is the pale pith cream
	CHECK(m.subsurfaceColor.texture != 0u);
	// the gltf split, thin sheet
	CHECK(m.subsurfaceScatterAnisotropy.r == doctest::Approx(1.0f));
	CHECK(m.geometryThinWalled.r == doctest::Approx(1.0f));
	// strength and colour are two different views of one image, so they must
	// resolve to two different bindless handles (linear for A, srgb for rgb)
	CHECK(m.subsurfaceWeight.texture != m.subsurfaceColor.texture);

	mor::scene_destroy(scene);
}

} // TEST_SUITE
