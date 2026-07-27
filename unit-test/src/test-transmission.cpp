#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

// tests for the openpbr transmission lobe evaluate/pdf trio,
// util-material-openpbr-transmission.glsl: the walter et al. 2007 rough
// dielectric btdf (openPbrTransmissionEvaluateF), its vndf refraction pdf
// (openPbrTransmissionPdf), the total-internal-reflection reflect-fallback
// pdf (openPbrTransmissionTirReflectPdf), and the cauchy/abbe dispersion
// helper (openPbrDispersionIorRgb). the port was verified byte-identical
// against cull's util-material-openpbr-transmission.glsl (only
// f32/f32v3 -> float/vec3 spellings differ) before these tests were
// written. sampler-vs-pdf statistics live in test-transmission-sampling.cpp;
// this file covers the closed-form/analytic side: cpu f64 goldens, the
// eta^2 walter reciprocity, thin-walled delta-lobe conventions, tint/depth
// semantics, dispersion goldens, a full materialx-assembled walter
// crosscheck, the matched-ior (etaI == etaT) thin-walled-delta-lobe
// regression, and the btdf disk / NaN sweep visualizations.
//
// conventions pinned throughout: canonical frame nor = (0,0,1), isotropic
// roughness (the lib's refraction path is isotropic-only for now),
// dispersedIor passed = specularIor (no wavelength lock), thin-walled off
// unless a test says otherwise.

namespace {

constexpr f64 kPi = 3.14159265358979323846;

// -----------------------------------------------------------------------------
// cpu f64 reference, transcribed 1:1 from the same formulas the shader
// ports (util-material-openpbr-transmission.glsl and the microfacet
// helpers it calls). not independent of the implementation -- it validates
// the gpu f32 evaluation against the intended math at double precision;
// independence comes from the materialx crosscheck below.
// -----------------------------------------------------------------------------

struct RefDir {
	f64 x;
	f64 y;
	f64 z;
};

RefDir ref_normalize(RefDir const v) {
	f64 const l = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	return { v.x / l, v.y / l, v.z / l };
}

f64 ref_dot(RefDir const a, RefDir const b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

f64 ref_fresnel_dielectric(f64 const cosTheta, f64 const eta) {
	f64 const cosTheta2 = cosTheta * cosTheta;
	f64 const sinTheta2 = 1.0 - cosTheta2;
	f64 const sinThetaT2 = sinTheta2 / (eta * eta);
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

f64 ref_ggx_d(f64 const dotNorH, f64 const alpha) {
	f64 const alpha2 = alpha * alpha;
	f64 const d = std::max(dotNorH * dotNorH * (alpha2 - 1.0) + 1.0, 1e-5);
	return alpha2 / (kPi * d * d);
}

f64 ref_smith_v(f64 const muI, f64 const muO, f64 const alpha) {
	if (muI <= 0.0 || muO <= 0.0) {
		return 0.0;
	}
	f64 const alpha2 = alpha * alpha;
	f64 const lambdaWi = (
		muO * std::sqrt(muI * muI * (1.0 - alpha2) + alpha2)
	);
	f64 const lambdaWo = (
		muI * std::sqrt(muO * muO * (1.0 - alpha2) + alpha2)
	);
	return 0.5 / std::max(lambdaWi + lambdaWo, 1e-7);
}

f64 ref_smith_g1(f64 const mu, f64 const alpha) {
	f64 const alpha2 = alpha * alpha;
	return (
		2.0 * mu / (mu + std::sqrt(alpha2 + (1.0 - alpha2) * mu * mu))
	);
}

struct RefTransmissionConfig {
	RefDir wi;
	RefDir wo;
	f64 ior;
	f64 roughness;
	bool isInsideMedium;
};

// scalar (white-tint, depth 0) transcription of openPbrTransmissionEvaluateF,
// non-thin-walled branch
f64 ref_transmission_f(RefTransmissionConfig const & c) {
	f64 const etaI = c.isInsideMedium ? c.ior : 1.0;
	f64 const etaT = c.isInsideMedium ? 1.0 : c.ior;
	RefDir const htRaw = {
		-(etaI * c.wi.x + etaT * c.wo.x),
		-(etaI * c.wi.y + etaT * c.wo.y),
		-(etaI * c.wi.z + etaT * c.wo.z),
	};
	RefDir const ht = ref_normalize(htRaw);
	RefDir const htOriented = ht.z < 0.0 ? RefDir{ -ht.x, -ht.y, -ht.z } : ht;
	f64 const dotNorWi = std::max(c.wi.z, 1e-5);
	f64 const dotNorWo = std::max(std::fabs(c.wo.z), 1e-5);
	f64 const dotHtWi = std::max(ref_dot(htOriented, c.wi), 0.0);
	f64 const dotHtWo = std::max(std::fabs(ref_dot(htOriented, c.wo)), 0.0);
	f64 const dotNorHt = std::max(htOriented.z, 0.0);
	f64 const alpha = c.roughness;
	f64 const fresnel = ref_fresnel_dielectric(dotHtWi, etaT / etaI);
	f64 const d = ref_ggx_d(dotNorHt, alpha);
	f64 const v = ref_smith_v(dotNorWi, dotNorWo, alpha);
	// length of the unnormalized half-vector sum, not a sum of two
	// independently-clamped cosines -- see the lib's denom comment
	f64 const denom = std::sqrt(
		htRaw.x * htRaw.x + htRaw.y * htRaw.y + htRaw.z * htRaw.z
	);
	return (
		(1.0 - fresnel) * d * v
		* 4.0 * dotHtWi * dotHtWo * etaT * etaT
		/ std::max(denom * denom, 1e-6)
	);
}

// transcription of openPbrTransmissionPdf, non-thin-walled branch
f64 ref_transmission_pdf(RefTransmissionConfig const & c) {
	f64 const etaI = c.isInsideMedium ? c.ior : 1.0;
	f64 const etaT = c.isInsideMedium ? 1.0 : c.ior;
	RefDir const htRaw = {
		-(etaI * c.wi.x + etaT * c.wo.x),
		-(etaI * c.wi.y + etaT * c.wo.y),
		-(etaI * c.wi.z + etaT * c.wo.z),
	};
	RefDir const ht = ref_normalize(htRaw);
	RefDir const htOriented = ht.z < 0.0 ? RefDir{ -ht.x, -ht.y, -ht.z } : ht;
	f64 const dotNorWi = std::max(c.wi.z, 1e-5);
	f64 const dotNorHt = std::max(htOriented.z, 0.0);
	f64 const dotHtWi = std::max(ref_dot(htOriented, c.wi), 0.0);
	f64 const dotHtWo = std::max(std::fabs(ref_dot(htOriented, c.wo)), 0.0);
	f64 const alpha = std::max(c.roughness, 1e-5);
	f64 const g1 = ref_smith_g1(dotNorWi, alpha);
	f64 const d = ref_ggx_d(dotNorHt, alpha);
	// see ref_transmission_f's denom comment
	f64 const denom = std::sqrt(
		htRaw.x * htRaw.x + htRaw.y * htRaw.y + htRaw.z * htRaw.z
	);
	return (
		g1 * d * dotHtWi / dotNorWi
		* etaT * etaT * dotHtWo
		/ std::max(denom * denom, 1e-6)
	);
}

// does (wi, wo, ior, isInsideMedium) reconstruct a half vector that
// genuinely underwent total internal reflection? mirrors the exact gate
// openPbrTransmissionTirReflectPdf now applies internally (glsl refract()'s
// own k < 0 discriminant, reconstructed here rather than called since
// refract() has no c++ stdlib equivalent) -- used to predict which of the
// fuzz test's random samples the lib should zero out
bool ref_tir_reflect_is_tir(
	RefDir const wi, RefDir const wo,
	f64 const ior, bool const isInsideMedium
) {
	RefDir const h = ref_normalize({ wi.x + wo.x, wi.y + wo.y, wi.z + wo.z });
	if (h.z <= 0.0 || ref_dot(wi, h) <= 0.0) {
		return false;
	}
	f64 const etaI = isInsideMedium ? ior : 1.0;
	f64 const etaT = isInsideMedium ? 1.0 : ior;
	f64 const eta = etaI / etaT;
	f64 const dotWiH = ref_dot(wi, h);
	f64 const k = 1.0 - eta * eta * (1.0 - dotWiH * dotWiH);
	return k < 0.0;
}

// transcription of openPbrDispersionIorRgb
void ref_dispersion_ior_rgb(
	f64 const ior, f64 const scale, f64 const abbe, f64 outRgb[3]
) {
	if (scale <= 0.0) {
		outRgb[0] = ior; outRgb[1] = ior; outRgb[2] = ior;
		return;
	}
	f64 const lambdaF = 0.4861;
	f64 const lambdaD = 0.5876;
	f64 const lambdaC = 0.6563;
	f64 const vd = abbe / scale;
	f64 const vdSafe = vd != 0.0 ? vd : 1e-4;
	f64 const b = (
		(ior - 1.0)
		/ (
			vdSafe
			* (1.0 / (lambdaF * lambdaF) - 1.0 / (lambdaC * lambdaC))
		)
	);
	f64 const a = ior - b / (lambdaD * lambdaD);
	outRgb[0] = a + b / (lambdaC * lambdaC);
	outRgb[1] = ior;
	outRgb[2] = a + b / (lambdaF * lambdaF);
}

// f64 glsl-refract; returns false on total internal reflection
bool ref_refract(
	RefDir const incident, RefDir const n, f64 const eta, RefDir & out
) {
	f64 const dotNi = ref_dot(n, incident);
	f64 const k = 1.0 - eta * eta * (1.0 - dotNi * dotNi);
	if (k < 0.0) {
		return false;
	}
	f64 const scale = eta * dotNi + std::sqrt(k);
	out = {
		eta * incident.x - scale * n.x,
		eta * incident.y - scale * n.y,
		eta * incident.z - scale * n.z,
	};
	return true;
}

// -----------------------------------------------------------------------------
// gpu dispatch helpers
// -----------------------------------------------------------------------------

struct FlatPush {
	u64 inVa;
	u64 outVa;
	u32 count;
};

// generic runner for the flat f32-in / f32-out evaluate shaders
std::vector<f32> run_flat_shader(
	char const * const shaderPath,
	std::vector<f32> const & inFlat,
	u32 const inStride,
	u32 const outStride
) {
	u32 const count = (u32)inFlat.size() / inStride;
	REQUIRE(inFlat.size() == (size_t)count * inStride);
	static constexpr u32 kLocalSize = 64;
	u32 const groups = (count + kLocalSize - 1) / kLocalSize;

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = shaderPath,
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto inBuf = vkof::buffer_create({
		.byteCount = inFlat.size() * sizeof(f32),
		.memory = vkof::BufferMemory::HostWritable,
	});
	vkof::buffer_upload({
		.buffer = inBuf, .byteOffset = 0,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(inFlat.data()),
			inFlat.size() * sizeof(f32)
		),
	});
	auto outBuf = vkof::buffer_create({
		.byteCount = (u64)count * outStride * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	FlatPush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
	};
	test::dispatch(pl, push, groups);

	auto out = test::readback<f32>(outBuf, 0, count * outStride);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pl);
	return out;
}

// transmission_evaluate.comp: 14 floats/sample in, 6 out
// (f.rgb, pdf, tirReflectPdf, unused)
void push_eval_sample(
	std::vector<f32> & flat,
	f32v3 const wi, f32v3 const wo,
	f32 const ior, f32 const roughness,
	f32 const thinWalled, f32 const isInsideMedium,
	f32v3 const tint = { 1.0f, 1.0f, 1.0f },
	f32 const depth = 0.0f
) {
	flat.push_back(wi.x); flat.push_back(wi.y); flat.push_back(wi.z);
	flat.push_back(wo.x); flat.push_back(wo.y); flat.push_back(wo.z);
	flat.push_back(ior);
	flat.push_back(roughness);
	flat.push_back(thinWalled);
	flat.push_back(isInsideMedium);
	flat.push_back(tint.x); flat.push_back(tint.y); flat.push_back(tint.z);
	flat.push_back(depth);
}

std::vector<f32> evaluate_transmission(std::vector<f32> const & flat) {
	return run_flat_shader(
		TEST_SHADER_DIR "transmission_evaluate.comp", flat, 14u, 6u
	);
}

f32v3 dir_from(f32 const cosTheta, f32 const phi) {
	f32 const sinTheta = (
		std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta))
	);
	return { sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta };
}

// deterministic lcg in [0, 1), matching the other test files
struct Rng {
	u32 seed;
	f32 next() {
		seed = seed * 747796405u + 2891336453u;
		return (f32)(seed >> 8) / (f32)(1u << 24);
	}
};

// cpu twin of the reachable-pair generator: pick an h by angle, refract
// wi about it in f64. returns false when that h undergoes tir
bool ref_reachable_wo(
	RefDir const wi, f64 const cosThetaH, f64 const phiH,
	f64 const ior, bool const isInsideMedium, RefDir & outWo
) {
	f64 const sinThetaH = std::sqrt(
		std::max(0.0, 1.0 - cosThetaH * cosThetaH)
	);
	RefDir const h = {
		sinThetaH * std::cos(phiH), sinThetaH * std::sin(phiH), cosThetaH
	};
	if (ref_dot(h, wi) <= 1e-4) {
		return false;
	}
	f64 const etaI = isInsideMedium ? ior : 1.0;
	f64 const etaT = isInsideMedium ? 1.0 : ior;
	RefDir refracted;
	if (
		!ref_refract(
			{ -wi.x, -wi.y, -wi.z }, h, etaI / etaT, refracted
		)
	) {
		return false;
	}
	outWo = ref_normalize(refracted);
	return true;
}

} // namespace

TEST_SUITE("[headless]") {

// -----------------------------------------------------------------------------
// golden values
// -----------------------------------------------------------------------------

TEST_CASE("transmission btdf: golden closed form at normal incidence (wi = n, wo = -n)") {
	// straight-through config: ht = +n exactly, so dotHtWi = 1 (raw
	// dot(h,wi)) and raw dot(h,wo) = -1 (h aligns with wi's side, wo is
	// transmitted to the opposite side) -- denom is the length of the
	// *unnormalized* sum etaI*wi + etaT*wo, which at this exact config is
	// |etaI - etaT| (the two terms partially cancel, not add), so:
	//   F = F0 = ((ior-1)/(ior+1))^2, D = 1/(pi alpha^2), V = 1/4,
	//   denom = (etaI - etaT)^2
	//   f = (1 - F0) / (pi alpha^2) * etaT^2 / (etaI - etaT)^2
	//   pdf = 1 / (pi alpha^2) * etaT^2 / (etaI - etaT)^2
	std::vector<f32> const iors = { 1.05f, 1.33f, 1.5f, 2.42f };
	std::vector<f32> const roughnesses = { 0.1f, 0.3f, 0.6f, 1.0f };
	std::vector<f32> flat;
	std::vector<f64> expectedF, expectedPdf;
	for (f32 const ior : iors) {
		for (f32 const rough : roughnesses) {
			for (u32 inside = 0u; inside < 2u; ++inside) {
				push_eval_sample(
					flat, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, -1.0f },
					ior, rough, 0.0f, (f32)inside
				);
				f64 const etaI = inside ? (f64)ior : 1.0;
				f64 const etaT = inside ? 1.0 : (f64)ior;
				// fresnel at normal incidence is symmetric in the ratio
				f64 const f0 = (
					((f64)ior - 1.0) * ((f64)ior - 1.0)
					/ (((f64)ior + 1.0) * ((f64)ior + 1.0))
				);
				f64 const d = 1.0 / (kPi * (f64)rough * (f64)rough);
				f64 const denom = (etaI - etaT) * (etaI - etaT);
				expectedF.push_back((1.0 - f0) * d * etaT * etaT / denom);
				expectedPdf.push_back(d * etaT * etaT / denom);
			}
		}
	}
	auto const out = evaluate_transmission(flat);
	for (u32 i = 0; i < expectedF.size(); ++i) {
		CAPTURE(i);
		CHECK(out[i*6+0] == doctest::Approx(expectedF[i]).epsilon(0.002));
		CHECK(out[i*6+3] == doctest::Approx(expectedPdf[i]).epsilon(0.002));
	}
}

TEST_CASE("transmission btdf: matches cpu f64 reference on reachable refractions (dense)") {
	// reachable (wi, wo) pairs generated by refracting about explicit half
	// vectors in f64; both entering and exiting the medium, ior and
	// roughness swept. checks f, pdf, and the tir-reflect pdf formula
	// itself (evaluated at an unrelated reflected-side wo built from the
	// same h)
	std::vector<f32> flat;
	std::vector<f64> expectedF, expectedPdf;
	Rng rng { 511u };
	u32 tirSkipped = 0u;
	for (u32 i = 0; i < 4096u; ++i) {
		f64 const cosThetaI = 0.05 + 0.95 * (f64)rng.next();
		f64 const sinThetaI = std::sqrt(1.0 - cosThetaI * cosThetaI);
		RefDir const wi = { sinThetaI, 0.0, cosThetaI };
		f64 const cosThetaH = 0.3 + 0.7 * (f64)rng.next();
		f64 const phiH = (f64)rng.next() * 2.0 * kPi;
		f64 const ior = 1.05 + 1.5 * (f64)rng.next();
		bool const inside = rng.next() > 0.5f;
		f64 const rough = 0.05 + 0.95 * (f64)rng.next();
		RefDir wo;
		if (!ref_reachable_wo(wi, cosThetaH, phiH, ior, inside, wo)) {
			tirSkipped++;
			continue;
		}
		RefTransmissionConfig const c = {
			wi, wo, ior, rough, inside
		};
		expectedF.push_back(ref_transmission_f(c));
		expectedPdf.push_back(ref_transmission_pdf(c));
		push_eval_sample(
			flat,
			{ (f32)wi.x, (f32)wi.y, (f32)wi.z },
			{ (f32)wo.x, (f32)wo.y, (f32)wo.z },
			(f32)ior, (f32)rough, 0.0f, inside ? 1.0f : 0.0f
		);
	}
	// tir skips happen only on inside-medium draws (half the draws), and
	// cosThetaH is swept across its full [0.3, 1.0] range including deep
	// past the critical angle for the sampled ior range, so a majority of
	// that inside-medium half legitimately skips; bound generously against
	// tir dominating the ENTERING half too (which would indicate a real
	// eta-direction bug), not against the natural exiting-side tir rate
	CHECK(tirSkipped < 3072u);
	auto const out = evaluate_transmission(flat);
	for (u32 i = 0; i < expectedF.size(); ++i) {
		CAPTURE(i);
		CHECK(out[i*6+0] == doctest::Approx(expectedF[i]).epsilon(0.01));
		CHECK(out[i*6+3] == doctest::Approx(expectedPdf[i]).epsilon(0.01));
	}
}

// openPbrTransmissionTirReflectPdf now gates on genuine TIR (ior and
// isInsideMedium do matter, unlike before this fix) and, when genuinely
// TIR, delegates entirely to utilMicrofacetGgxBoundedReflectPdfAniso --
// the same bounded reflect pdf primitive already verified independently
// in the dielectric-specular / sampling-pdf milestones. re-deriving that
// primitive's full algorithm as a second cpu f64 reference here would
// duplicate, not independently check, that existing verification, so
// this test's scope is the gate: exactly zero when genuinely not TIR,
// finite and positive when genuinely TIR
TEST_CASE("transmission tir-reflect pdf: zero exactly off the TIR domain (fuzz)") {
	std::vector<f32> flat;
	std::vector<bool> expectedIsTir;
	Rng rng { 613u };
	for (u32 i = 0; i < 2048u; ++i) {
		f32 const muI = 0.05f + 0.95f * rng.next();
		f32 const muO = 0.05f + 0.95f * rng.next();
		f32v3 const wi = dir_from(muI, 0.0f);
		f32v3 const wo = dir_from(muO, rng.next() * 6.2831853f);
		f32 const rough = 0.02f + 0.98f * rng.next();
		f32 const ior = 1.1f + rng.next();
		bool const isInsideMedium = rng.next() > 0.5f;
		push_eval_sample(
			flat, wi, wo, ior, rough, 0.0f, isInsideMedium ? 1.0f : 0.0f
		);
		expectedIsTir.push_back(
			ref_tir_reflect_is_tir(
				{ wi.x, wi.y, wi.z }, { wo.x, wo.y, wo.z }, ior, isInsideMedium
			)
		);
	}
	auto const out = evaluate_transmission(flat);
	for (u32 i = 0; i < expectedIsTir.size(); ++i) {
		CAPTURE(i);
		CAPTURE(expectedIsTir[i]);
		f32 const tirPdf = out[i*6+4];
		CHECK(std::isfinite(tirPdf));
		if (expectedIsTir[i]) {
			CHECK(tirPdf > 0.0f);
		} else {
			CHECK(tirPdf == 0.0f);
		}
	}
}

// -----------------------------------------------------------------------------
// physical properties
// -----------------------------------------------------------------------------

TEST_CASE("transmission btdf: walter eta^2 reciprocity on reachable refractions") {
	// walter et al. 2007 eq 8: f_t(wi, wo) / etaT^2 is symmetric, i.e.
	//   f_enter(wi, wo) / ior^2 == f_exit(-wo, -wi) / 1^2
	// (the reverse path enters from inside the medium, so its config
	// mirrors through the surface with isInsideMedium = true). checked on
	// f64-generated reachable pairs; V and D are swap-symmetric and the
	// fresnel factors agree analytically, so this pins the etaT^2 scaling
	// and the direction conventions all at once
	std::vector<f32> forward, reversed;
	Rng rng { 88u };
	u32 count = 0u;
	std::vector<f64> iors;
	for (u32 i = 0; i < 2048u && count < 1024u; ++i) {
		f64 const cosThetaI = 0.1 + 0.88 * (f64)rng.next();
		f64 const sinThetaI = std::sqrt(1.0 - cosThetaI * cosThetaI);
		RefDir const wi = { sinThetaI, 0.0, cosThetaI };
		f64 const cosThetaH = 0.5 + 0.5 * (f64)rng.next();
		f64 const phiH = (f64)rng.next() * 2.0 * kPi;
		f64 const ior = 1.05 + 1.3 * (f64)rng.next();
		f64 const rough = 0.05 + 0.9 * (f64)rng.next();
		RefDir wo;
		if (!ref_reachable_wo(wi, cosThetaH, phiH, ior, false, wo)) {
			continue;
		}
		count++;
		iors.push_back(ior);
		push_eval_sample(
			forward,
			{ (f32)wi.x, (f32)wi.y, (f32)wi.z },
			{ (f32)wo.x, (f32)wo.y, (f32)wo.z },
			(f32)ior, (f32)rough, 0.0f, 0.0f
		);
		push_eval_sample(
			reversed,
			{ (f32)-wo.x, (f32)-wo.y, (f32)-wo.z },
			{ (f32)-wi.x, (f32)-wi.y, (f32)-wi.z },
			(f32)ior, (f32)rough, 0.0f, 1.0f
		);
	}
	REQUIRE(count == 1024u);
	auto const fwd = evaluate_transmission(forward);
	auto const rev = evaluate_transmission(reversed);
	for (u32 i = 0; i < count; ++i) {
		CAPTURE(i);
		f64 const lhs = (f64)fwd[i*6+0] / (iors[i] * iors[i]);
		f64 const rhs = (f64)rev[i*6+0];
		CHECK(lhs == doctest::Approx(rhs).epsilon(0.005));
	}
}

TEST_CASE("transmission btdf: non-negative and finite over the production domain (fuzz)") {
	// production domain: wi above the horizon, wo anywhere, ior in
	// [1.01, 3], any roughness, both medium sides, tint/depth mixed. the
	// matched-ior (ior = 1) degenerate point is excluded here -- it gets
	// its own dedicated probes below
	std::vector<f32> flat;
	Rng rng { 999u };
	for (u32 i = 0; i < 8192u; ++i) {
		f32 const muI = (i % 7u == 0u) ? rng.next() * 0.01f : rng.next();
		f32 const muO = rng.next() * 2.0f - 1.0f;
		push_eval_sample(
			flat,
			dir_from(std::min(muI, 0.99995f), rng.next() * 6.2831853f),
			dir_from(muO, rng.next() * 6.2831853f),
			1.01f + rng.next() * 1.99f,
			rng.next(),
			(i % 11u == 0u) ? 1.0f : 0.0f,
			rng.next() > 0.5f ? 1.0f : 0.0f,
			{ rng.next(), rng.next(), rng.next() },
			(i % 3u == 0u) ? rng.next() : 0.0f
		);
	}
	auto const out = evaluate_transmission(flat);
	for (u32 i = 0; i < out.size() / 6u; ++i) {
		CAPTURE(i);
		CHECK(std::isfinite(out[i*6+0]));
		CHECK(std::isfinite(out[i*6+3]));
		CHECK(std::isfinite(out[i*6+4]));
		CHECK(out[i*6+0] >= 0.0f);
		CHECK(out[i*6+3] >= 0.0f);
		CHECK(out[i*6+4] >= 0.0f);
	}
}

TEST_CASE("transmission btdf: thin-walled delta lobe conventions") {
	// thin-walled: f = tint * (1 - F(dotNorWi, etaRel)) / |dotNorWo| for
	// ANY wo (the caller only ever pairs it with wo = -wi), and the pdf is
	// exactly 1 (the lobe's full weight at its single delta direction)
	std::vector<f32> flat;
	std::vector<f64> expectedF;
	std::vector<f32> const iors = { 1.1f, 1.5f, 2.0f };
	std::vector<f32> const mus = { 0.05f, 0.3f, 0.7f, 1.0f };
	for (f32 const ior : iors) {
		for (f32 const mu : mus) {
			f32v3 const wi = dir_from(mu, 0.0f);
			f32v3 const wo = { -wi.x, -wi.y, -wi.z };
			for (u32 inside = 0u; inside < 2u; ++inside) {
				push_eval_sample(
					flat, wi, wo, ior, 0.4f, 1.0f, (f32)inside,
					{ 0.8f, 0.6f, 0.4f }, 0.0f
				);
				f64 const etaI = inside ? (f64)ior : 1.0;
				f64 const etaT = inside ? 1.0 : (f64)ior;
				f64 const fres = ref_fresnel_dielectric(
					std::max((f64)mu, 1e-5), etaT / etaI
				);
				expectedF.push_back(
					0.8 * (1.0 - fres) / std::max((f64)mu, 1e-5)
				);
			}
		}
	}
	auto const out = evaluate_transmission(flat);
	for (u32 i = 0; i < expectedF.size(); ++i) {
		CAPTURE(i);
		CHECK(out[i*6+0] == doctest::Approx(expectedF[i]).epsilon(0.002));
		CHECK(out[i*6+3] == 1.0f);
	}
}

TEST_CASE("transmission btdf: tint applies at depth 0 and is ignored at depth > 0") {
	// transmissionColor is a constant fresnel tint only when
	// transmissionDepth = 0; with depth > 0 the color comes from
	// beer-lambert absorption elsewhere and the btdf itself goes white
	f32v3 const wi = dir_from(0.8f, 0.0f);
	RefDir wo64;
	REQUIRE(
		ref_reachable_wo({ wi.x, wi.y, wi.z }, 0.95, 0.0, 1.5, false, wo64)
	);
	f32v3 const wo = { (f32)wo64.x, (f32)wo64.y, (f32)wo64.z };
	f32v3 const tint = { 0.9f, 0.5f, 0.2f };
	std::vector<f32> flat;
	push_eval_sample(flat, wi, wo, 1.5f, 0.3f, 0.0f, 0.0f, tint, 0.0f);
	push_eval_sample(
		flat, wi, wo, 1.5f, 0.3f, 0.0f, 0.0f,
		{ 1.0f, 1.0f, 1.0f }, 0.0f
	);
	push_eval_sample(flat, wi, wo, 1.5f, 0.3f, 0.0f, 0.0f, tint, 0.5f);
	auto const out = evaluate_transmission(flat);
	// tinted = white * tint, componentwise
	CHECK(out[0*6+0] == doctest::Approx(out[1*6+0] * 0.9f).epsilon(0.001));
	CHECK(out[0*6+1] == doctest::Approx(out[1*6+1] * 0.5f).epsilon(0.001));
	CHECK(out[0*6+2] == doctest::Approx(out[1*6+2] * 0.2f).epsilon(0.001));
	// depth > 0 ignores the tint entirely
	CHECK(out[2*6+0] == doctest::Approx(out[1*6+0]).epsilon(0.001));
	CHECK(out[2*6+1] == doctest::Approx(out[1*6+1]).epsilon(0.001));
	CHECK(out[2*6+2] == doctest::Approx(out[1*6+2]).epsilon(0.001));
}

// -----------------------------------------------------------------------------
// dispersion
// -----------------------------------------------------------------------------

TEST_CASE("transmission dispersion: matches cpu cauchy/abbe reference and orders r < g < b") {
	std::vector<f32> flat;
	std::vector<f64> expected;
	std::vector<f32> const iors = { 1.2f, 1.5f, 1.8f, 2.42f };
	std::vector<f32> const scales = { 0.0f, 0.1f, 0.5f, 1.0f, 2.0f };
	std::vector<f32> const abbes = { 9.0f, 20.0f, 55.0f, 91.0f };
	for (f32 const ior : iors) {
		for (f32 const scale : scales) {
			for (f32 const abbe : abbes) {
				flat.push_back(ior);
				flat.push_back(scale);
				flat.push_back(abbe);
				f64 rgb[3];
				ref_dispersion_ior_rgb(ior, scale, abbe, rgb);
				expected.push_back(rgb[0]);
				expected.push_back(rgb[1]);
				expected.push_back(rgb[2]);
			}
		}
	}
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "transmission_dispersion_evaluate.comp", flat, 3u, 3u
	);
	for (u32 i = 0; i < out.size() / 3u; ++i) {
		CAPTURE(i);
		CHECK(out[i*3+0] == doctest::Approx(expected[i*3+0]).epsilon(0.001));
		CHECK(out[i*3+1] == doctest::Approx(expected[i*3+1]).epsilon(0.001));
		CHECK(out[i*3+2] == doctest::Approx(expected[i*3+2]).epsilon(0.001));
		// green channel is the authored ior exactly (a, b are solved so
		// n(lambda_d) == specularIor)
		f32 const ior = flat[i*3+0];
		f32 const scale = flat[i*3+1];
		CHECK(out[i*3+1] == ior);
		if (scale > 0.0f && ior > 1.0f) {
			// normal dispersion: shorter wavelengths bend more,
			// n(F/blue) > n(d/green) > n(C/red)
			CHECK(out[i*3+2] > out[i*3+1]);
			CHECK(out[i*3+1] > out[i*3+0]);
		} else if (scale == 0.0f) {
			CHECK(out[i*3+0] == ior);
			CHECK(out[i*3+2] == ior);
		}
	}
}

TEST_CASE("transmission dispersion: zero abbe number with positive scale stays finite (guard)") {
	// the vdSafe floor exists exactly for this malformed input; a NaN here
	// would poison every downstream ior
	std::vector<f32> const flat = { 1.5f, 1.0f, 0.0f };
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "transmission_dispersion_evaluate.comp", flat, 3u, 3u
	);
	CHECK(std::isfinite(out[0]));
	CHECK(std::isfinite(out[1]));
	CHECK(std::isfinite(out[2]));
	CHECK(out[1] == 1.5f);
}

// -----------------------------------------------------------------------------
// materialx independent cross-check (walter assembly from mx primitives;
// see transmission_crosscheck.comp for the setup)
// -----------------------------------------------------------------------------

namespace {

// transmission_crosscheck.comp: 8 floats/sample in, 8 out
void push_crosscheck_sample(
	std::vector<f32> & flat,
	f32v3 const wi, f32 const ior, f32 const roughness,
	f32 const isInsideMedium, f32v2 const xi
) {
	flat.push_back(wi.x); flat.push_back(wi.y); flat.push_back(wi.z);
	flat.push_back(ior);
	flat.push_back(roughness);
	flat.push_back(isInsideMedium);
	flat.push_back(xi.x); flat.push_back(xi.y);
}

} // namespace

TEST_CASE("transmission btdf: matches materialx walter assembly (grid)") {
	std::vector<f32> flat;
	std::vector<f32> const iors = { 1.1f, 1.33f, 1.5f, 2.0f };
	std::vector<f32> const roughnesses = { 0.05f, 0.2f, 0.5f, 1.0f };
	std::vector<f32> const cosThetas = { 0.98f, 0.7f, 0.4f, 0.1f };
	Rng rng { 2718u };
	for (f32 const ior : iors) {
		for (f32 const rough : roughnesses) {
			for (f32 const ct : cosThetas) {
				for (u32 inside = 0u; inside < 2u; ++inside) {
					for (u32 s = 0u; s < 8u; ++s) {
						push_crosscheck_sample(
							flat, dir_from(ct, 0.7f), ior, rough,
							(f32)inside, { rng.next(), rng.next() }
						);
					}
				}
			}
		}
	}
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "transmission_crosscheck.comp", flat, 8u, 8u
	);
	u32 tirSkipped = 0u;
	f64 maxRelF = 0.0;
	f64 maxRelPdf = 0.0;
	for (u32 i = 0; i < out.size() / 8u; ++i) {
		if (out[i*8+4] > 0.5f) {
			tirSkipped++;
			continue;
		}
		CAPTURE(i);
		CHECK(out[i*8+0] == doctest::Approx(out[i*8+1]).epsilon(0.005));
		CHECK(out[i*8+2] == doctest::Approx(out[i*8+3]).epsilon(0.005));
		f64 const relF = (
			std::fabs((f64)out[i*8+0] - (f64)out[i*8+1])
			/ std::max((f64)out[i*8+1], 1e-9)
		);
		f64 const relPdf = (
			std::fabs((f64)out[i*8+2] - (f64)out[i*8+3])
			/ std::max((f64)out[i*8+3], 1e-9)
		);
		maxRelF = std::max(maxRelF, relF);
		maxRelPdf = std::max(maxRelPdf, relPdf);
	}
	// tir skips only occur on the inside-medium half at steep half vectors
	CHECK(tirSkipped < out.size() / 8u / 3u);
	MESSAGE(
		"materialx walter crosscheck max relative diff: f ", maxRelF,
		", pdf ", maxRelPdf, " (", tirSkipped, " tir-skipped)"
	);
}

TEST_CASE("transmission btdf: matches materialx walter assembly (fuzz)") {
	std::vector<f32> flat;
	Rng rng { 31415u };
	for (u32 i = 0; i < 4096u; ++i) {
		push_crosscheck_sample(
			flat,
			dir_from(0.02f + 0.97f * rng.next(), rng.next() * 6.2831853f),
			1.02f + rng.next() * 1.5f,
			0.02f + rng.next() * 0.98f,
			rng.next() > 0.5f ? 1.0f : 0.0f,
			{ rng.next(), rng.next() }
		);
	}
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "transmission_crosscheck.comp", flat, 8u, 8u
	);
	u32 tirSkipped = 0u;
	for (u32 i = 0; i < out.size() / 8u; ++i) {
		if (out[i*8+4] > 0.5f) {
			tirSkipped++;
			continue;
		}
		CAPTURE(i);
		CHECK(out[i*8+0] == doctest::Approx(out[i*8+1]).epsilon(0.01));
		CHECK(out[i*8+2] == doctest::Approx(out[i*8+3]).epsilon(0.01));
	}
	CHECK(tirSkipped < 2048u);
}

// -----------------------------------------------------------------------------
// degenerate-point probe, now a fixed-behavior regression test: matched ior
// (etaI == etaT) shares the thin-walled delta lobe rather than falling
// through to the continuous vndf btdf, so it no longer hits the
// -(etaI*wi + etaT*wo) == 0 degenerate normalize() (reported and fixed
// 2026-07-15, "option A" -- reuse the thin-walled branch rather than guard
// the NaN, since an index-matched interface is physically the same
// undeviated full-transmission case: F is identically 0 at eta == 1 and
// refract() with eta == 1 always returns wo == -wi)
// -----------------------------------------------------------------------------

TEST_CASE("transmission btdf: matched ior (ior == 1) takes the thin-walled delta lobe") {
	// f == tint / dotNorWo (F == 0 identically at eta == 1) and pdf == 1,
	// exactly matching the thin-walled closed form -- white tint here
	// makes f == 1 / mu
	std::vector<f32> flat;
	std::vector<f32> const mus = { 1.0f, 0.7f, 0.3f, 0.05f };
	for (f32 const mu : mus) {
		f32v3 const wi = dir_from(mu, 0.0f);
		f32v3 const wo = { -wi.x, -wi.y, -wi.z };
		push_eval_sample(flat, wi, wo, 1.0f, 0.3f, 0.0f, 0.0f);
	}
	auto const out = evaluate_transmission(flat);
	for (u32 i = 0; i < mus.size(); ++i) {
		CAPTURE(mus[i]);
		CHECK(std::isfinite(out[i*6+0]));
		CHECK(out[i*6+0] == doctest::Approx(1.0f / mus[i]).epsilon(1e-4));
		CHECK(out[i*6+3] == 1.0f);
	}
}

TEST_CASE("transmission btdf: near-matched ior takes the continuous path (regression fence)") {
	// just off etaI == etaT the continuous vndf btdf branch takes over
	// again (not the thin-walled-style delta lobe); pins that the
	// etaI == etaT fast-exit is exact-equality only and doesn't leak into
	// this neighborhood
	std::vector<f32> flat;
	std::vector<f32> const iors = { 1.0001f, 1.001f, 1.01f };
	for (f32 const ior : iors) {
		f32v3 const wi = dir_from(0.7f, 0.0f);
		// near-straight-through direction, the closest reachable analog
		// of the delta config above
		// zero-initialized: ref_reachable_wo always fully writes outWo
		// before returning true (and the REQUIRE below aborts on false
		// before wo64 is ever read), but gcc's flow analysis can't always
		// prove that across the call boundary -- silences a spurious
		// -Wmaybe-uninitialized/-Werror build failure
		RefDir wo64 {};
		REQUIRE(
			ref_reachable_wo(
				{ wi.x, wi.y, wi.z }, 0.999, 0.0, (f64)ior, false, wo64
			)
		);
		push_eval_sample(
			flat, wi,
			{ (f32)wo64.x, (f32)wo64.y, (f32)wo64.z },
			ior, 0.3f, 0.0f, 0.0f
		);
	}
	auto const out = evaluate_transmission(flat);
	for (u32 i = 0; i < iors.size(); ++i) {
		CAPTURE(iors[i]);
		CHECK(std::isfinite(out[i*6+0]));
		CHECK(std::isfinite(out[i*6+3]));
		CHECK(out[i*6+3] > 0.0f);
	}
}

// -----------------------------------------------------------------------------
// visual representation
// -----------------------------------------------------------------------------

namespace {

void write_transmission_disk(
	f32 const roughness, f32 const ior, char const * const path,
	f32 const tonemapHeadroom
) {
	// btdf * |cos| over the TRANSMITTED (lower) hemisphere projected to
	// the tangent disk, wi fixed at 45 degrees in +x, ior'd glass; the
	// refraction lobe sits opposite wi, compressed toward the normal by
	// snell bending. outside the disk renders as background black,
	// non-finite values as pure red
	constexpr u32 kSize = 512;
	f32 const muI = 0.7071f;
	f32v3 const wi = { std::sqrt(1.0f - muI * muI), 0.0f, muI };

	std::vector<f32> flat;
	std::vector<bool> valid((size_t)kSize * kSize, false);
	flat.reserve((size_t)kSize * kSize * 14);
	for (u32 y = 0; y < kSize; ++y) {
		f32 const wy = 2.0f * (f32)y / (f32)(kSize - 1) - 1.0f;
		for (u32 x = 0; x < kSize; ++x) {
			f32 const wx = 2.0f * (f32)x / (f32)(kSize - 1) - 1.0f;
			f32 const r2 = wx * wx + wy * wy;
			bool const inside = r2 <= 1.0f;
			valid[y * kSize + x] = inside;
			f32v3 const wo = {
				wx, wy, inside ? -std::sqrt(1.0f - r2) : -1.0f
			};
			push_eval_sample(flat, wi, wo, ior, roughness, 0.0f, 0.0f);
		}
	}
	auto const out = evaluate_transmission(flat);

	std::vector<f32> r((size_t)kSize * kSize, 0.0f);
	std::vector<f32> g((size_t)kSize * kSize, 0.0f);
	std::vector<f32> b((size_t)kSize * kSize, 0.0f);
	for (u32 i = 0; i < (u32)(kSize * kSize); ++i) {
		if (!valid[i]) { continue; }
		f32 const wx = 2.0f * (f32)(i % kSize) / (f32)(kSize - 1) - 1.0f;
		f32 const wy = 2.0f * (f32)(i / kSize) / (f32)(kSize - 1) - 1.0f;
		f32 const cosO = std::sqrt(
			std::max(0.0f, 1.0f - wx * wx - wy * wy)
		);
		f32 const v = out[i*6+0] * cosO * 3.14159265f / tonemapHeadroom;
		r[i] = v; g[i] = v; b[i] = v;
	}

	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		r, g, b, kSize, kSize, path, &nanCount
	);
	CHECK(ok);
	CAPTURE(nanCount);
	CHECK(nanCount == 0);
}

} // namespace

TEST_CASE("transmission btdf: transmitted disk, smooth glass") {
	write_transmission_disk(
		0.05f, 1.5f, TRANSMISSION_OUTPUT_DIR "transmission_disk_smooth.png",
		12.0f
	);
}

TEST_CASE("transmission btdf: transmitted disk, medium glass") {
	write_transmission_disk(
		0.2f, 1.5f, TRANSMISSION_OUTPUT_DIR "transmission_disk_medium.png",
		4.0f
	);
}

TEST_CASE("transmission btdf: transmitted disk, rough glass") {
	write_transmission_disk(
		0.6f, 1.5f, TRANSMISSION_OUTPUT_DIR "transmission_disk_rough.png",
		2.0f
	);
}

TEST_CASE("transmission btdf: transmitted disk, dense flint (high ior)") {
	write_transmission_disk(
		0.2f, 2.0f, TRANSMISSION_OUTPUT_DIR "transmission_disk_dense.png",
		4.0f
	);
}

TEST_CASE("transmission btdf: eval NaN sweep heatmaps (incidence x outgoing)") {
	// in-plane (phi = pi) slice over (cosThetaI, cosThetaO), f and pdf
	// evaluated everywhere including unreachable configurations; NaN/Inf
	// renders pure red. the matched-ior config used to document a known
	// wo == -wi anti-diagonal degeneracy (fixed 2026-07-15: etaI == etaT
	// now takes the same thin-walled delta lobe as geometryThinWalled),
	// so all three configs are expected clean here
	constexpr u32 kSize = 256;
	struct SweepConfig {
		f32 ior;
		f32 inside;
		char const * path;
	};
	std::vector<SweepConfig> const configs = {
		{
			1.5f, 0.0f,
			TRANSMISSION_OUTPUT_DIR "transmission_eval_sweep_enter.png",
		},
		{
			1.5f, 1.0f,
			TRANSMISSION_OUTPUT_DIR "transmission_eval_sweep_exit.png",
		},
		{
			1.0f, 0.0f,
			TRANSMISSION_OUTPUT_DIR "transmission_eval_sweep_matched_ior.png",
		},
	};
	for (auto const & cfg : configs) {
		std::vector<f32> flat;
		for (u32 y = 0; y < kSize; ++y) {
			// wo spans the full sphere
			f32 const muO = 1.0f - 2.0f * (f32)y / (f32)(kSize - 1);
			for (u32 x = 0; x < kSize; ++x) {
				f32 const muI = (
					0.005f + 0.995f * (f32)x / (f32)(kSize - 1)
				);
				push_eval_sample(
					flat, dir_from(muI, 0.0f),
					dir_from(muO, 3.14159265f),
					cfg.ior, 0.4f, 0.0f, cfg.inside
				);
			}
		}
		auto const out = evaluate_transmission(flat);
		std::vector<f32> r((size_t)kSize * kSize);
		std::vector<f32> g((size_t)kSize * kSize);
		std::vector<f32> b((size_t)kSize * kSize);
		u32 nonFinite = 0u;
		for (u32 i = 0; i < (u32)(kSize * kSize); ++i) {
			f32 const f = out[i*6+0];
			f32 const pdf = out[i*6+3];
			if (!std::isfinite(f) || !std::isfinite(pdf)) {
				nonFinite++;
				r[i] = std::nanf("");
				g[i] = std::nanf("");
				b[i] = std::nanf("");
				continue;
			}
			// f in red/green, pdf in blue, both range-compressed
			r[i] = f / (f + 1.0f);
			g[i] = f / (f + 1.0f);
			b[i] = pdf / (pdf + 1.0f);
		}
		u32 nanCount = 0;
		bool const ok = test::write_heatmap_png(
			r, g, b, kSize, kSize, cfg.path, &nanCount
		);
		CHECK(ok);
		CAPTURE(cfg.ior);
		CAPTURE(cfg.inside);
		MESSAGE(
			"eval sweep ior=", cfg.ior, " inside=", cfg.inside,
			": nonFinite texels = ", nonFinite, " / ", kSize * kSize
		);
		CHECK(nonFinite == 0u);
	}
}

TEST_CASE("transmission dispersion: ior spread heatmap (abbe x scale)") {
	// visual map of the cauchy fit: x = abbe number in [9, 91], y =
	// dispersion scale in [0, 2], channels show (iorRgb - 1) / 1.5 --
	// the red/blue split widens toward low abbe and high scale, green
	// stays constant at the authored ior. red pixels would mean NaN
	constexpr u32 kSize = 128;
	std::vector<f32> flat;
	for (u32 y = 0; y < kSize; ++y) {
		f32 const scale = 2.0f * (f32)y / (f32)(kSize - 1);
		for (u32 x = 0; x < kSize; ++x) {
			f32 const abbe = 9.0f + 82.0f * (f32)x / (f32)(kSize - 1);
			flat.push_back(1.5f);
			flat.push_back(scale);
			flat.push_back(abbe);
		}
	}
	auto const out = run_flat_shader(
		TEST_SHADER_DIR "transmission_dispersion_evaluate.comp", flat, 3u, 3u
	);
	std::vector<f32> r((size_t)kSize * kSize);
	std::vector<f32> g((size_t)kSize * kSize);
	std::vector<f32> b((size_t)kSize * kSize);
	for (u32 i = 0; i < (u32)(kSize * kSize); ++i) {
		r[i] = (out[i*3+0] - 1.0f) / 1.5f;
		g[i] = (out[i*3+1] - 1.0f) / 1.5f;
		b[i] = (out[i*3+2] - 1.0f) / 1.5f;
	}
	u32 nanCount = 0;
	bool const ok = test::write_heatmap_png(
		r, g, b, kSize, kSize,
		TRANSMISSION_OUTPUT_DIR "transmission_dispersion_spread.png",
		&nanCount
	);
	CHECK(ok);
	CHECK(nanCount == 0);
}

} // TEST_SUITE("[headless]")
