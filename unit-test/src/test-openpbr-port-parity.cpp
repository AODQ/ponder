#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <ponder/zeltner-tables.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// parity of the new-core openpbr port (lib/ponder/shaders, module bsdf)
// against the old pipeline (shaders-old), plus materialx where it has an
// equivalent. the old functions carry the full test history in
// test-fresnel / test-microfacet / test-microfacet-sampling /
// test-energy-compensation / test-roughness, so parity transfers it

namespace
{

// mirrors ParityFunction in openpbr-port-parity.slang
enum class ParityFunction : u32 {
	fresnelDielectric = 0u,
	fresnelMetallic = 1u,
	ggxDistributionAniso = 2u,
	smithVisibilityAniso = 3u,
	smithG1Aniso = 4u,
	sampleGgxBoundedWoAniso = 5u,
	sampleGgxVndfAniso = 6u,
	ggxBoundedReflectPdfAniso = 7u,
	ggxDirectionalAlbedo = 8u,
	dielectricEnergyCompensate = 9u,
	dielectricAlbedo = 10u,
	roughnessAlpha = 11u,
	coatRoughenedRoughness = 12u,
	coatAffectedIor = 13u,
	effectiveIor = 14u,
	specularEffectiveIor = 15u,
	iorFromF0 = 16u,
	halfVectorLocal = 17u,
	fresnelDielectricPhase = 18u,
	fresnelConductorPhase = 19u,
	evaluateXyzSensitivity = 20u,
	thinfilmFresnel = 21u,
	dielectricSpecularEvaluateF = 22u,
	fresnelMetallicEvaluateF = 23u,
	orenNayerEnergyCompensate = 24u,
	glossyDiffuseEvaluateF = 25u,
	thinWalledSubsurfaceEvaluateF = 26u,
	thinWalledSubsurfacePdf = 27u,
	glossyDiffusePdf = 28u,
	zeltnerFuzzLookup = 29u,
	fuzzEvaluateF = 30u,
	fuzzPdf = 31u,
	fuzzSampleWo = 32u,
	coatEvaluateF = 33u,
	transmissionEvaluateF = 34u,
	transmissionPdf = 35u,
	transmissionTirReflectPdf = 36u,
	dispersionIorRgb = 37u,
	subsurfaceDiffusionReflectance = 38u,
	subsurfaceInvertDiffusionReflectance = 39u,
	subsurfaceSigmaAbsorption = 40u,
	subsurfaceSigmaScattering = 41u,
	subsurfacePhaseEvalHg = 42u,
	subsurfacePhaseSampleHgCos = 43u,
	subsurfacePhaseSampleWo = 44u,
	dwivediDiffusionLength = 45u,
	dwivediPhaseEval = 46u,
	dwivediPhaseSampleCos = 47u,
	subsurfaceEntrySampleWo = 48u,
	subsurfaceEntryPdf = 49u,
};

struct ParityPush {
	u64 inVa;
	u64 outVa;
	u32 count;
	u32 function;
	u32 zeltnerLtcParamHandle;
	u32 pad0;
};

// nine f32v4 of inputs per sample
struct ParityInput {
	f32 a[4];
	f32 b[4];
	f32 c[4];
	f32 d[4];
	f32 e[4];
	f32 f[4];
	f32 g[4];
	f32 h[4];
	f32 k[4];
};
static_assert(sizeof(ParityInput) == 144u);

// three f32v4 of outputs per sample: new, old, materialx
struct ParityOutput {
	f32 next[4];
	f32 old[4];
	f32 mx[4];
};
static_assert(sizeof(ParityOutput) == 48u);

static constexpr u32 skLocalSize = 64u;
static constexpr u32 skFuzzCount = 4096u;
// matches skMinRoughness in the openpbr port
static constexpr f32 skMinAlpha = 1e-3f;
// same code on both sides; only compiler reassociation may differ
static constexpr f32 skParityRelTol = 1e-5f;
static constexpr f32 skParityAbsTol = 1e-7f;
static constexpr f32 skPi = 3.14159265358979323846f;

std::vector<ParityOutput> parity_evaluate(
	ParityFunction const function,
	std::vector<ParityInput> const & inputs,
	u32 const zeltnerLtcParamHandle = 0u
) {
	u32 const count = (u32)inputs.size();
	u32 const groups = (count + skLocalSize - 1u) / skLocalSize;

	char const * const includePaths[] = {
		PONDER_SHADER_DIR, PONDER_SHADER_NEW_DIR,
	};
	char const * const defines[] = { "PONDER_SPECTRUM_RGB" };
	vkof::Pipeline const pipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = TEST_SHADER_DIR "slang/entry/openpbr-port-parity.slang",
			.defines = srat::slice<char const * const>(defines, 1u),
			.includePaths = srat::slice<char const * const>(includePaths, 2u),
		})
	);
	REQUIRE(pipeline.id != 0u);

	vkof::Buffer const inBuf = (
		vkof::buffer_create({
			.byteCount = count * sizeof(ParityInput),
			.memory = vkof::BufferMemory::HostWritable,
		})
	);
	vkof::buffer_upload({
		.buffer = inBuf,
		.byteOffset = 0u,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(inputs.data()),
			count * sizeof(ParityInput)
		),
	});
	vkof::Buffer const outBuf = (
		vkof::buffer_create({
			.byteCount = count * sizeof(ParityOutput),
			.memory = vkof::BufferMemory::DeviceOnly,
		})
	);

	ParityPush const push {
		.inVa = vkof::buffer_virtual_address(inBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = count,
		.function = (u32)function,
		.zeltnerLtcParamHandle = zeltnerLtcParamHandle,
		.pad0 = 0u,
	};
	test::dispatch(pipeline, push, groups);
	std::vector<ParityOutput> const outputs = (
		test::readback<ParityOutput>(outBuf, 0u, count)
	);

	vkof::buffer_destroy(inBuf);
	vkof::buffer_destroy(outBuf);
	vkof::pipeline_destroy(pipeline);
	return outputs;
}

// relative-or-absolute closeness; nan only matches nan
bool values_match(
	f32 const x,
	f32 const y,
	f32 const relTol,
	f32 const absTol
) {
	if (std::isnan(x) || std::isnan(y)) {
		return std::isnan(x) && std::isnan(y);
	}
	if (std::isinf(x) || std::isinf(y)) {
		return x == y;
	}
	f32 const diff = std::fabs(x - y);
	f32 const scale = std::max(std::fabs(x), std::fabs(y));
	return diff <= absTol || diff <= relTol * scale;
}

void print_input(ParityInput const & in) {
	std::printf(
		(
			"    a=(%g, %g, %g, %g) b=(%g, %g, %g, %g)"
			" c=(%g, %g, %g, %g) d=(%g, %g, %g, %g)"
			" e=(%g, %g, %g, %g) f=(%g, %g, %g, %g)"
			" g=(%g, %g, %g, %g) h=(%g, %g, %g, %g) k=(%g, %g, %g, %g)\n"
		),
		in.a[0], in.a[1], in.a[2], in.a[3],
		in.b[0], in.b[1], in.b[2], in.b[3],
		in.c[0], in.c[1], in.c[2], in.c[3],
		in.d[0], in.d[1], in.d[2], in.d[3],
		in.e[0], in.e[1], in.e[2], in.e[3],
		in.f[0], in.f[1], in.f[2], in.f[3],
		in.g[0], in.g[1], in.g[2], in.g[3],
		in.h[0], in.h[1], in.h[2], in.h[3],
		in.k[0], in.k[1], in.k[2], in.k[3]
	);
}

// compares new against old (or materialx) over the first laneCount lanes
u32 count_mismatches(
	char const * const label,
	std::vector<ParityInput> const & inputs,
	std::vector<ParityOutput> const & outputs,
	u32 const laneCount,
	bool const againstMaterialx,
	f32 const relTol,
	f32 const absTol
) {
	u32 mismatches = 0u;
	for (size_t i = 0u; i < outputs.size(); ++i) {
		ParityOutput const & out = outputs[i];
		f32 const * const reference = againstMaterialx ? out.mx : out.old;
		for (u32 lane = 0u; lane < laneCount; ++lane) {
			if (values_match(out.next[lane], reference[lane], relTol, absTol)) {
				continue;
			}
			if (mismatches < 4u) {
				std::printf(
					"[%s] sample %zu lane %u: new %.9g vs %s %.9g\n",
					label, i, lane, out.next[lane],
					againstMaterialx ? "materialx" : "old", reference[lane]
				);
				print_input(inputs[i]);
			}
			++mismatches;
			break;
		}
	}
	return mismatches;
}

void check_parity(
	char const * const label,
	ParityFunction const function,
	std::vector<ParityInput> const & inputs,
	u32 const laneCount,
	u32 const zeltnerLtcParamHandle = 0u
) {
	std::vector<ParityOutput> const outputs = (
		parity_evaluate(function, inputs, zeltnerLtcParamHandle)
	);
	u32 const mismatches = (
		count_mismatches(
			label, inputs, outputs, laneCount,
			/*againstMaterialx=*/false, skParityRelTol, skParityAbsTol
		)
	);
	CAPTURE(label);
	CHECK(mismatches == 0u);
}

// -- input generation

struct Rng {
	std::mt19937 engine;

	f32 uniform(f32 const lo, f32 const hi) {
		std::uniform_real_distribution<f32> distribution(lo, hi);
		return distribution(engine);
	}

	// log-uniform in [skMinAlpha, 1], so low roughness is well covered
	f32 alpha() {
		return std::exp(uniform(std::log(skMinAlpha), 0.0f));
	}

	void unitSphere(f32 * const out) {
		f32 const z = uniform(-1.0f, 1.0f);
		f32 const phi = uniform(0.0f, 2.0f * skPi);
		f32 const r = std::sqrt(std::max(0.0f, 1.0f - z * z));
		out[0] = r * std::cos(phi);
		out[1] = r * std::sin(phi);
		out[2] = z;
	}

	void unitHemisphere(f32 * const out) {
		unitSphere(out);
		out[2] = std::fabs(out[2]);
	}
};

Rng rng_create(u32 const seed) {
	return Rng { .engine = std::mt19937(seed) };
}

ParityInput input_zero() {
	return ParityInput {};
}

} // namespace

// -----------------------------------------------------------------------------
// -- fresnel
// -----------------------------------------------------------------------------

TEST_CASE("openpbr port parity: fresnel dielectric") {
	Rng rng = rng_create(1u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		in.a[0] = rng.uniform(0.0f, 1.0f);
		in.a[1] = rng.uniform(0.3f, 3.0f);
		inputs.emplace_back(in);
	}
	// grazing, normal, matched ior, tir edge
	for (f32 const cosTheta : { 0.0f, 1e-4f, 0.5f, 1.0f }) {
		for (f32 const eta : { 0.3f, 1.0f / 1.5f, 1.0f, 1.5f, 3.0f }) {
			// exact grazing + matched ior is pinned separately below
			if (cosTheta == 0.0f && eta == 1.0f) {
				continue;
			}
			ParityInput in = input_zero();
			in.a[0] = cosTheta;
			in.a[1] = eta;
			inputs.emplace_back(in);
		}
	}
	check_parity("fresnel dielectric", ParityFunction::fresnelDielectric, inputs, 1u);
}

TEST_CASE("openpbr port: fresnel dielectric is 1 at exact grazing with matched ior") {
	// a genuine discontinuity: 0 along η = 1, 1 along cosθ = 0. the port
	// takes the grazing limit (sinθt² >= 1 is tir); the old code took 0
	ParityInput in = input_zero();
	in.a[0] = 0.0f;
	in.a[1] = 1.0f;
	std::vector<ParityOutput> const outputs = (
		parity_evaluate(ParityFunction::fresnelDielectric, { in })
	);
	CHECK(outputs[0].next[0] == 1.0f);
}

TEST_CASE("openpbr port parity: fresnel dielectric matches materialx") {
	// same exclusion as test-fresnel: exact grazing is a shared 0/0
	Rng rng = rng_create(2u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		in.a[0] = rng.uniform(0.01f, 1.0f);
		in.a[1] = rng.uniform(0.3f, 3.0f);
		inputs.emplace_back(in);
	}
	std::vector<ParityOutput> const outputs = (
		parity_evaluate(ParityFunction::fresnelDielectric, inputs)
	);
	u32 const mismatches = (
		count_mismatches(
			"fresnel dielectric vs materialx", inputs, outputs, 1u,
			/*againstMaterialx=*/true, 1e-3f, 1e-6f
		)
	);
	CHECK(mismatches == 0u);
}

TEST_CASE("openpbr port parity: fresnel metallic (f82-tint), per channel") {
	Rng rng = rng_create(3u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		for (u32 k = 0u; k < 3u; ++k) {
			in.a[k] = rng.uniform(0.0f, 1.0f);
			in.b[k] = rng.uniform(0.0f, 1.0f);
		}
		in.a[3] = rng.uniform(0.0f, 1.0f);
		in.b[3] = rng.uniform(0.0f, 1.0f);
		inputs.emplace_back(in);
	}
	check_parity("fresnel metallic", ParityFunction::fresnelMetallic, inputs, 3u);
}

// -----------------------------------------------------------------------------
// -- microfacet primitives
// -----------------------------------------------------------------------------

TEST_CASE("openpbr port parity: half vector") {
	Rng rng = rng_create(4u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		rng.unitSphere(in.a);
		rng.unitSphere(in.b);
		inputs.emplace_back(in);
	}
	// antiparallel: the degenerate case
	ParityInput degenerate = input_zero();
	degenerate.a[2] = 1.0f;
	degenerate.b[2] = -1.0f;
	inputs.emplace_back(degenerate);
	check_parity("half vector", ParityFunction::halfVectorLocal, inputs, 4u);
}

TEST_CASE("openpbr port parity: ggx distribution (aniso), and materialx") {
	Rng rng = rng_create(5u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		rng.unitHemisphere(in.a);
		in.b[0] = rng.alpha();
		in.b[1] = rng.alpha();
		inputs.emplace_back(in);
	}
	check_parity("ggx D", ParityFunction::ggxDistributionAniso, inputs, 1u);

	std::vector<ParityOutput> const outputs = (
		parity_evaluate(ParityFunction::ggxDistributionAniso, inputs)
	);
	u32 const mismatches = (
		count_mismatches(
			"ggx D vs materialx", inputs, outputs, 1u,
			/*againstMaterialx=*/true, 1e-4f, 1e-6f
		)
	);
	CHECK(mismatches == 0u);
}

TEST_CASE("openpbr port: ggx distribution peak is 1/(π alphaX alphaY) down to min roughness") {
	// the old isotropic D floors its denominator at 1e-5 and loses the peak
	// below α ≈ 3.2e-3; the anisotropic one used by every lobe must not
	std::vector<ParityInput> inputs;
	for (f32 const alphaX : { skMinAlpha, 2e-3f, 5e-3f, 1e-2f, 0.1f, 1.0f }) {
		for (f32 const alphaY : { skMinAlpha, 3e-3f, 0.05f, 1.0f }) {
			ParityInput in = input_zero();
			in.a[2] = 1.0f;
			in.b[0] = alphaX;
			in.b[1] = alphaY;
			inputs.emplace_back(in);
		}
	}
	std::vector<ParityOutput> const outputs = (
		parity_evaluate(ParityFunction::ggxDistributionAniso, inputs)
	);
	for (size_t i = 0u; i < inputs.size(); ++i) {
		f32 const expected = 1.0f / (skPi * inputs[i].b[0] * inputs[i].b[1]);
		CAPTURE(inputs[i].b[0]);
		CAPTURE(inputs[i].b[1]);
		CHECK(outputs[i].next[0] == doctest::Approx(expected).epsilon(1e-4));
	}
}

TEST_CASE("openpbr port parity: smith visibility (aniso)") {
	Rng rng = rng_create(6u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		// full sphere: the backfacing early-out is part of the contract
		rng.unitSphere(in.a);
		rng.unitSphere(in.b);
		in.c[0] = rng.alpha();
		in.c[1] = rng.alpha();
		inputs.emplace_back(in);
	}
	check_parity("smith V", ParityFunction::smithVisibilityAniso, inputs, 1u);
}

TEST_CASE("openpbr port parity: smith g1 (aniso), and materialx when isotropic") {
	Rng rng = rng_create(7u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		rng.unitHemisphere(in.a);
		// materialx's g1 is isotropic, so the shared inputs are too
		f32 const alpha = rng.alpha();
		in.b[0] = alpha;
		in.b[1] = alpha;
		inputs.emplace_back(in);
	}
	check_parity("smith G1", ParityFunction::smithG1Aniso, inputs, 1u);

	std::vector<ParityOutput> const outputs = (
		parity_evaluate(ParityFunction::smithG1Aniso, inputs)
	);
	u32 const mismatches = (
		count_mismatches(
			"smith G1 vs materialx", inputs, outputs, 1u,
			/*againstMaterialx=*/true, 1e-4f, 1e-6f
		)
	);
	CHECK(mismatches == 0u);
}

// -----------------------------------------------------------------------------
// -- microfacet sampling and pdf
// -----------------------------------------------------------------------------

TEST_CASE("openpbr port parity: bounded vndf reflection sampler (aniso)") {
	Rng rng = rng_create(8u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		// full sphere: the sampler has a separate i_z <= 0 branch
		rng.unitSphere(in.a);
		in.b[0] = rng.alpha();
		in.b[1] = rng.alpha();
		in.c[0] = rng.uniform(0.0f, 1.0f);
		in.c[1] = rng.uniform(0.0f, 1.0f);
		inputs.emplace_back(in);
	}
	// ωi antiparallel to the normal: the macro-normal fallback
	ParityInput degenerate = input_zero();
	degenerate.a[2] = -1.0f;
	degenerate.b[0] = 0.3f;
	degenerate.b[1] = 0.3f;
	degenerate.c[0] = 0.25f;
	degenerate.c[1] = 0.5f;
	inputs.emplace_back(degenerate);
	check_parity("bounded sampler", ParityFunction::sampleGgxBoundedWoAniso, inputs, 3u);
}

TEST_CASE("openpbr port parity: vndf microfacet-normal sampler (aniso)") {
	Rng rng = rng_create(9u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		rng.unitHemisphere(in.a);
		in.b[0] = rng.alpha();
		in.b[1] = rng.alpha();
		in.c[0] = rng.uniform(0.0f, 1.0f);
		in.c[1] = rng.uniform(0.0f, 1.0f);
		inputs.emplace_back(in);
	}
	// normal incidence: the T1 fallback branch
	ParityInput normal = input_zero();
	normal.a[2] = 1.0f;
	normal.b[0] = 0.2f;
	normal.b[1] = 0.2f;
	normal.c[0] = 0.4f;
	normal.c[1] = 0.6f;
	inputs.emplace_back(normal);
	check_parity("vndf sampler", ParityFunction::sampleGgxVndfAniso, inputs, 3u);
}

TEST_CASE("openpbr port parity: bounded vndf reflection pdf (aniso)") {
	Rng rng = rng_create(10u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		// both full sphere: exercises the cap-cut tests and the i_z < 0 form
		rng.unitSphere(in.a);
		rng.unitSphere(in.b);
		in.c[0] = rng.alpha();
		in.c[1] = rng.alpha();
		inputs.emplace_back(in);
	}
	check_parity("bounded pdf", ParityFunction::ggxBoundedReflectPdfAniso, inputs, 1u);
}

// -----------------------------------------------------------------------------
// -- energy compensation
// -----------------------------------------------------------------------------

TEST_CASE("openpbr port parity: ggx directional albedo, and materialx") {
	Rng rng = rng_create(11u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		in.a[0] = rng.uniform(0.0f, 1.0f);
		in.a[1] = rng.uniform(0.0f, 1.0f);
		in.a[2] = rng.uniform(0.0f, 1.0f);
		in.a[3] = rng.uniform(0.0f, 1.0f);
		inputs.emplace_back(in);
	}
	check_parity("directional albedo", ParityFunction::ggxDirectionalAlbedo, inputs, 1u);

	std::vector<ParityOutput> const outputs = (
		parity_evaluate(ParityFunction::ggxDirectionalAlbedo, inputs)
	);
	u32 const mismatches = (
		count_mismatches(
			"directional albedo vs materialx", inputs, outputs, 1u,
			/*againstMaterialx=*/true, 1e-4f, 1e-6f
		)
	);
	CHECK(mismatches == 0u);
}

TEST_CASE("openpbr port parity: dielectric energy compensation and albedo") {
	Rng rng = rng_create(12u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		in.a[0] = rng.uniform(0.0f, 1.0f);
		in.a[1] = rng.uniform(0.0f, 1.0f);
		in.a[2] = rng.uniform(0.0f, 1.0f);
		inputs.emplace_back(in);
	}
	// grazing / rough corner where the fit reaches 0 and the 1e-4 floor acts
	ParityInput corner = input_zero();
	corner.a[0] = 0.0f;
	corner.a[1] = 1.0f;
	corner.a[2] = 0.04f;
	inputs.emplace_back(corner);
	check_parity(
		"energy compensate", ParityFunction::dielectricEnergyCompensate, inputs, 1u
	);
	check_parity("dielectric albedo", ParityFunction::dielectricAlbedo, inputs, 1u);
}

// -----------------------------------------------------------------------------
// -- roughness and ior
// -----------------------------------------------------------------------------

TEST_CASE("openpbr port parity: roughness alpha and coat roughening") {
	Rng rng = rng_create(13u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		in.a[0] = rng.uniform(0.0f, 1.0f);
		in.a[1] = rng.uniform(0.0f, 1.0f);
		in.a[2] = rng.uniform(0.0f, 1.0f);
		inputs.emplace_back(in);
	}
	// anisotropy 1 collapses one axis onto the floor
	ParityInput full = input_zero();
	full.a[0] = 0.5f;
	full.a[1] = 1.0f;
	inputs.emplace_back(full);
	check_parity("roughness alpha", ParityFunction::roughnessAlpha, inputs, 2u);
	check_parity(
		"coat roughened roughness", ParityFunction::coatRoughenedRoughness, inputs, 1u
	);
}

TEST_CASE("openpbr port parity: ior helpers") {
	Rng rng = rng_create(14u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		// sub-unity ratios exercise the sign handling in the effective ior
		in.a[0] = rng.uniform(0.5f, 3.0f);
		in.a[1] = rng.uniform(1.0f, 3.0f);
		in.a[2] = rng.uniform(0.0f, 1.0f);
		in.a[3] = rng.uniform(0.0f, 1.0f);
		inputs.emplace_back(in);
	}
	// no coat: early return; matched ior: sign(0)
	ParityInput noCoat = input_zero();
	noCoat.a[0] = 1.5f;
	noCoat.a[1] = 1.6f;
	noCoat.a[3] = 1.0f;
	inputs.emplace_back(noCoat);
	ParityInput matched = input_zero();
	matched.a[0] = 1.0f;
	matched.a[1] = 1.0f;
	matched.a[2] = 1.0f;
	matched.a[3] = 1.0f;
	inputs.emplace_back(matched);

	check_parity("coat affected ior", ParityFunction::coatAffectedIor, inputs, 1u);

	// effective ior reads (ior, weight) from a.xy
	std::vector<ParityInput> effectiveInputs;
	for (ParityInput const & in : inputs) {
		ParityInput effective = input_zero();
		effective.a[0] = in.a[0];
		effective.a[1] = in.a[3];
		effectiveInputs.emplace_back(effective);
	}
	check_parity("effective ior", ParityFunction::effectiveIor, effectiveInputs, 1u);

	check_parity(
		"specular effective ior", ParityFunction::specularEffectiveIor, inputs, 1u
	);
}

TEST_CASE("openpbr port parity: ior from f0, per channel") {
	Rng rng = rng_create(15u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		in.a[0] = rng.uniform(0.0f, 1.0f);
		in.a[1] = rng.uniform(0.0f, 1.0f);
		in.a[2] = rng.uniform(0.0f, 1.0f);
		inputs.emplace_back(in);
	}
	// both clamp ends
	ParityInput ends = input_zero();
	ends.a[0] = 0.0f;
	ends.a[1] = 1.0f;
	ends.a[2] = 0.9999f;
	inputs.emplace_back(ends);
	check_parity("ior from f0", ParityFunction::iorFromF0, inputs, 3u);
}

// -----------------------------------------------------------------------------
// -- thin film
// -----------------------------------------------------------------------------

TEST_CASE("openpbr port parity: thin-film interface phase functions") {
	Rng rng = rng_create(16u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		f32 const cosThetaI = rng.uniform(0.0f, 1.0f);
		f32 const etaI = rng.uniform(1.0f, 2.5f);
		f32 const etaT = rng.uniform(0.3f, 3.0f);
		f32 const sinThetaT2 = (
			(etaI / etaT) * (etaI / etaT) * (1.0f - cosThetaI * cosThetaI)
		);
		in.a[0] = cosThetaI;
		in.a[1] = std::sqrt(std::max(0.0f, 1.0f - sinThetaT2));
		in.a[2] = etaI;
		in.a[3] = etaT;
		// κ = 0 exercises the dielectric fallback inside the conductor phase
		in.b[0] = (i % 4u == 0u) ? 0.0f : rng.uniform(0.0f, 5.0f);
		inputs.emplace_back(in);
	}
	check_parity(
		"dielectric phase", ParityFunction::fresnelDielectricPhase, inputs, 4u
	);
	check_parity(
		"conductor phase", ParityFunction::fresnelConductorPhase, inputs, 4u
	);
}

TEST_CASE("openpbr port parity: thin-film xyz sensitivity") {
	Rng rng = rng_create(17u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		// opd in micrometers; a few microns covers every visible order
		in.a[0] = rng.uniform(0.0f, 6.0f);
		in.a[1] = rng.uniform(0.0f, 4.0f * skPi);
		inputs.emplace_back(in);
	}
	check_parity(
		"xyz sensitivity", ParityFunction::evaluateXyzSensitivity, inputs, 3u
	);
}

// no thin-film fresnel parity: the lib fixed C_m = (R - T_{12}^2) to
// (R - T_{12}) and the old pipeline keeps the bug; test-thinfilm.cpp checks
// the lib against the f64 reference and materialx instead

// -----------------------------------------------------------------------------
// -- dielectric specular lobe
// -----------------------------------------------------------------------------

TEST_CASE("openpbr port parity: dielectric specular lobe, per channel") {
	Rng rng = rng_create(19u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		// full sphere for ωo: the lobe must return 0 below the horizon
		rng.unitHemisphere(in.a);
		rng.unitSphere(in.b);
		in.a[3] = rng.uniform(0.5f, 2.5f);
		in.c[0] = rng.uniform(0.0f, 1.0f);
		in.c[1] = rng.uniform(0.0f, 1.0f);
		in.c[2] = rng.uniform(0.0f, 1.0f);
		in.c[3] = rng.uniform(0.0f, 1.0f);
		in.d[0] = rng.uniform(0.0f, 2.0f * skPi);
		// thin film off: the old pipeline's thin film keeps a fixed bug
		in.d[1] = 0.0f;
		in.d[2] = rng.uniform(1.0f, 3.0f);
		in.d[3] = rng.uniform(0.0f, 2.0f);
		inputs.emplace_back(in);
	}
	check_parity(
		"dielectric specular", ParityFunction::dielectricSpecularEvaluateF, inputs, 3u
	);
}

// -----------------------------------------------------------------------------
// -- metallic lobe
// -----------------------------------------------------------------------------

TEST_CASE("openpbr port parity: metallic lobe (f82-tint), per channel") {
	Rng rng = rng_create(20u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		// full sphere for ωo: below the horizon V is zero
		rng.unitHemisphere(in.a);
		rng.unitSphere(in.b);
		in.a[3] = rng.uniform(0.0f, 1.0f);
		// an eighth non-metal: the early return
		in.b[3] = (i % 8u == 0u) ? 0.0f : rng.uniform(0.0f, 1.0f);
		for (u32 k = 0u; k < 3u; ++k) {
			in.c[k] = rng.uniform(0.0f, 1.0f);
			in.d[k] = rng.uniform(0.0f, 1.0f);
		}
		in.c[3] = rng.uniform(0.0f, 1.0f);
		in.d[3] = rng.uniform(0.0f, 1.0f);
		in.e[0] = rng.uniform(0.0f, 1.0f);
		in.e[1] = rng.uniform(0.0f, 1.0f);
		in.e[2] = rng.uniform(0.0f, 1.0f);
		in.e[3] = rng.uniform(0.0f, 2.0f * skPi);
		// thin film off: the old pipeline's thin film keeps a fixed bug
		in.f[0] = 0.0f;
		in.f[1] = rng.uniform(1.0f, 3.0f);
		in.f[2] = rng.uniform(0.0f, 2.0f);
		inputs.emplace_back(in);
	}
	check_parity(
		"metallic lobe", ParityFunction::fresnelMetallicEvaluateF, inputs, 3u
	);
}

// -----------------------------------------------------------------------------
// -- eon diffuse
// -----------------------------------------------------------------------------

TEST_CASE("openpbr port parity: eon energy compensation, and materialx exact") {
	Rng rng = rng_create(21u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		// the exact form is 0/0 at μ = 0; same grid floor as test-eon-diffuse
		in.a[0] = rng.uniform(0.005f, 1.0f);
		in.a[1] = rng.uniform(0.0f, 1.0f);
		inputs.emplace_back(in);
	}
	check_parity(
		"eon energy compensate", ParityFunction::orenNayerEnergyCompensate, inputs, 1u
	);

	// the portsmouth fit is sub-percent; same bound as test-eon-diffuse
	std::vector<ParityOutput> const outputs = (
		parity_evaluate(ParityFunction::orenNayerEnergyCompensate, inputs)
	);
	u32 const mismatches = (
		count_mismatches(
			"eon energy compensate vs materialx", inputs, outputs, 1u,
			/*againstMaterialx=*/true, 0.0f, 0.015f
		)
	);
	CHECK(mismatches == 0u);
}

TEST_CASE("openpbr port parity: eon glossy diffuse, per channel, and materialx") {
	Rng rng = rng_create(22u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		// μ >= 0.02 on both sides; same floor as test-eon-diffuse
		do { rng.unitHemisphere(in.a); } while (in.a[2] < 0.02f);
		do { rng.unitHemisphere(in.b); } while (in.b[2] < 0.02f);
		in.a[3] = rng.uniform(0.25f, 1.0f);
		in.b[3] = rng.uniform(0.0f, 1.0f);
		in.c[0] = rng.uniform(0.0f, 1.0f);
		in.c[1] = rng.uniform(0.0f, 1.0f);
		in.c[2] = rng.uniform(0.0f, 1.0f);
		inputs.emplace_back(in);
	}
	check_parity(
		"eon glossy diffuse", ParityFunction::glossyDiffuseEvaluateF, inputs, 3u
	);

	// the only modeling delta is the polynomial vs exact E_FON; same
	// absolute bound as test-eon-diffuse (~1.5% of the lambert level)
	std::vector<ParityOutput> const outputs = (
		parity_evaluate(ParityFunction::glossyDiffuseEvaluateF, inputs)
	);
	u32 const mismatches = (
		count_mismatches(
			"eon glossy diffuse vs materialx", inputs, outputs, 3u,
			/*againstMaterialx=*/true, 0.0f, 0.005f
		)
	);
	CHECK(mismatches == 0u);
}

TEST_CASE("openpbr port parity: thin-walled subsurface sheet, per channel") {
	Rng rng = rng_create(23u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		// ωo on the full sphere: reflection and transmission halves
		rng.unitHemisphere(in.a);
		rng.unitSphere(in.b);
		in.a[3] = rng.uniform(0.0f, 1.0f);
		in.b[3] = rng.uniform(-1.0f, 1.0f);
		in.c[0] = rng.uniform(0.0f, 1.0f);
		in.c[1] = rng.uniform(0.0f, 1.0f);
		in.c[2] = rng.uniform(0.0f, 1.0f);
		inputs.emplace_back(in);
	}
	check_parity(
		"thin-walled subsurface",
		ParityFunction::thinWalledSubsurfaceEvaluateF,
		inputs,
		3u
	);
}

TEST_CASE("openpbr port parity: diffuse and thin-walled pdfs") {
	Rng rng = rng_create(24u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		in.a[0] = rng.uniform(-1.0f, 1.0f);
		// out-of-range anisotropy exercises the clamp
		in.a[1] = rng.uniform(-1.5f, 1.5f);
		inputs.emplace_back(in);
	}
	check_parity(
		"thin-walled pdf", ParityFunction::thinWalledSubsurfacePdf, inputs, 1u
	);
	check_parity("glossy diffuse pdf", ParityFunction::glossyDiffusePdf, inputs, 1u);
}

// -----------------------------------------------------------------------------
// -- fuzz
// -----------------------------------------------------------------------------

TEST_CASE("openpbr port: zeltner fuzz table lookup matches the reference fetch") {
	ponder::ZeltnerTables tables = ponder::zeltner_tables_create();
	Rng rng = rng_create(25u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		in.a[0] = rng.uniform(0.0f, 1.0f);
		in.a[1] = rng.uniform(0.0f, 1.0f);
		inputs.emplace_back(in);
	}
	// table nodes: cosθ = i/31, α = j/31
	for (u32 m = 0u; m < 31u; ++m) {
		for (u32 r = 0u; r < 31u; ++r) {
			ParityInput in = input_zero();
			in.a[0] = (f32)m / 31.0f;
			in.a[1] = (f32)r / 31.0f;
			inputs.emplace_back(in);
		}
	}
	std::vector<ParityOutput> const outputs = (
		parity_evaluate(
			ParityFunction::zeltnerFuzzLookup, inputs, tables.zeltnerLtcParamHandle
		)
	);
	u32 const mismatches = (
		count_mismatches(
			"zeltner lookup vs reference", inputs, outputs, 3u,
			/*againstMaterialx=*/true, 1e-4f, 1e-6f
		)
	);
	CHECK(mismatches == 0u);
	ponder::zeltner_tables_destroy(tables);
}

TEST_CASE("openpbr port: fuzz lobe matches the reference, per channel") {
	ponder::ZeltnerTables tables = ponder::zeltner_tables_create();
	Rng rng = rng_create(26u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		rng.unitHemisphere(in.a);
		rng.unitSphere(in.b);
		// an eighth without fuzz: the early return
		in.a[3] = (i % 8u == 0u) ? 0.0f : rng.uniform(0.0f, 1.0f);
		in.b[3] = rng.uniform(0.0f, 1.0f);
		for (u32 c = 0u; c < 3u; ++c) {
			in.c[c] = rng.uniform(0.0f, 1.0f);
			in.d[c] = rng.uniform(0.0f, 1.0f);
		}
		inputs.emplace_back(in);
	}
	std::vector<ParityOutput> const outputs = (
		parity_evaluate(
			ParityFunction::fuzzEvaluateF, inputs, tables.zeltnerLtcParamHandle
		)
	);
	u32 const mismatches = (
		count_mismatches(
			"fuzz lobe vs reference", inputs, outputs, 3u,
			/*againstMaterialx=*/true, 1e-4f, 1e-6f
		)
	);
	CHECK(mismatches == 0u);
	ponder::zeltner_tables_destroy(tables);
}

TEST_CASE("openpbr port parity: fuzz pdf and sampler") {
	Rng rng = rng_create(27u);
	std::vector<ParityInput> pdfInputs;
	std::vector<ParityInput> sampleInputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		f32 const fuzzA = rng.uniform(0.05f, 1.0f);
		f32 const fuzzB = rng.uniform(-0.5f, 0.5f);
		ParityInput pdf = input_zero();
		rng.unitHemisphere(pdf.a);
		rng.unitSphere(pdf.b);
		pdf.a[3] = fuzzA;
		pdf.b[3] = fuzzB;
		pdfInputs.emplace_back(pdf);
		ParityInput sample = input_zero();
		rng.unitHemisphere(sample.a);
		sample.a[3] = fuzzA;
		sample.b[0] = rng.uniform(0.0f, 1.0f);
		sample.b[1] = rng.uniform(0.0f, 1.0f);
		sample.b[3] = fuzzB;
		sampleInputs.emplace_back(sample);
	}
	// ωi along the normal: the arbitrary-azimuth frame fallback
	ParityInput normalPdf = input_zero();
	normalPdf.a[2] = 1.0f;
	normalPdf.a[3] = 0.5f;
	normalPdf.b[0] = 0.6f;
	normalPdf.b[2] = 0.8f;
	normalPdf.b[3] = 0.2f;
	pdfInputs.emplace_back(normalPdf);
	ParityInput normalSample = input_zero();
	normalSample.a[2] = 1.0f;
	normalSample.a[3] = 0.5f;
	normalSample.b[0] = 0.3f;
	normalSample.b[1] = 0.7f;
	normalSample.b[3] = 0.2f;
	sampleInputs.emplace_back(normalSample);
	check_parity("fuzz pdf", ParityFunction::fuzzPdf, pdfInputs, 1u);
	check_parity("fuzz sampler", ParityFunction::fuzzSampleWo, sampleInputs, 3u);
}

// -----------------------------------------------------------------------------
// -- coat
// -----------------------------------------------------------------------------

TEST_CASE("openpbr port parity: coat layering, per channel") {
	Rng rng = rng_create(28u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		rng.unitHemisphere(in.a);
		rng.unitSphere(in.b);
		// a quarter seen from inside the medium: inverted coat ior
		in.a[3] = (i % 4u == 0u) ? 1.0f : 0.0f;
		// an eighth with a matched coat ior: the no-coat early return
		in.b[3] = (i % 8u == 1u) ? 1.0f : rng.uniform(1.0f, 2.5f);
		for (u32 c = 0u; c < 3u; ++c) {
			in.c[c] = rng.uniform(0.0f, 1.0f);
			in.d[c] = rng.uniform(0.0f, 1.0f);
			in.e[c] = rng.uniform(0.0f, 1.0f);
			in.f[c] = rng.uniform(0.0f, 1.0f);
			in.g[c] = rng.uniform(0.0f, 1.0f);
		}
		in.c[3] = rng.uniform(0.0f, 1.0f);
		in.d[3] = rng.uniform(0.0f, 1.0f);
		in.e[3] = rng.uniform(0.0f, 1.0f);
		in.f[3] = rng.uniform(0.0f, 1.0f);
		in.g[3] = rng.uniform(1.0f, 2.5f);
		in.h[0] = rng.uniform(0.0f, 1.0f);
		in.h[1] = rng.uniform(0.0f, 1.0f);
		in.h[2] = rng.uniform(0.0f, 1.0f);
		in.h[3] = rng.uniform(0.0f, 1.0f);
		in.k[0] = rng.uniform(0.0f, 1.0f);
		in.k[1] = rng.uniform(0.0f, 1.0f);
		inputs.emplace_back(in);
	}
	check_parity("coat layering", ParityFunction::coatEvaluateF, inputs, 3u);
}

// -----------------------------------------------------------------------------
// -- transmission
// -----------------------------------------------------------------------------

TEST_CASE("openpbr port parity: transmission btdf, pdf and tir reflect pdf") {
	Rng rng = rng_create(29u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		rng.unitHemisphere(in.a);
		rng.unitSphere(in.b);
		// half seen from inside the medium
		in.a[3] = (i % 2u == 0u) ? 1.0f : 0.0f;
		// an eighth with a matched ior: the delta branch
		in.b[3] = (i % 8u == 3u) ? 1.0f : rng.uniform(1.05f, 2.5f);
		in.c[0] = rng.uniform(0.0f, 1.0f);
		in.c[1] = rng.uniform(0.0f, 1.0f);
		in.c[2] = rng.uniform(0.0f, 1.0f);
		// half with depth: no fresnel tint
		in.c[3] = (i % 4u < 2u) ? 0.0f : rng.uniform(0.1f, 2.0f);
		in.d[0] = rng.uniform(0.0f, 1.0f);
		in.d[1] = rng.uniform(0.0f, 1.0f);
		in.d[2] = rng.uniform(0.0f, 2.0f * skPi);
		// an eighth thin-walled: the delta branch
		in.d[3] = (i % 8u == 5u) ? 1.0f : 0.0f;
		inputs.emplace_back(in);
	}
	// exact delta direction, ωo = -ωi, through a thin shell
	ParityInput delta = input_zero();
	delta.a[2] = 1.0f;
	delta.b[2] = -1.0f;
	delta.b[3] = 1.5f;
	delta.c[0] = 0.8f;
	delta.c[1] = 0.6f;
	delta.c[2] = 0.4f;
	delta.d[0] = 0.3f;
	delta.d[3] = 1.0f;
	inputs.emplace_back(delta);
	check_parity("transmission btdf", ParityFunction::transmissionEvaluateF, inputs, 3u);
	check_parity("transmission pdf", ParityFunction::transmissionPdf, inputs, 1u);
	check_parity(
		"transmission tir reflect pdf",
		ParityFunction::transmissionTirReflectPdf,
		inputs,
		1u
	);
}

TEST_CASE("openpbr port parity: cauchy dispersion ior at the fraunhofer lines") {
	Rng rng = rng_create(30u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		in.a[0] = rng.uniform(1.0f, 3.0f);
		in.a[1] = rng.uniform(1.0f, 3.0f);
		in.a[2] = rng.uniform(0.0f, 1.0f);
		in.a[3] = rng.uniform(0.0f, 1.0f);
		// an eighth without dispersion: the early return
		in.b[0] = (i % 8u == 0u) ? 0.0f : rng.uniform(0.0f, 1.0f);
		in.b[1] = rng.uniform(9.0f, 91.0f);
		inputs.emplace_back(in);
	}
	check_parity("dispersion ior", ParityFunction::dispersionIorRgb, inputs, 3u);
}

// -----------------------------------------------------------------------------
// -- subsurface
// -----------------------------------------------------------------------------

TEST_CASE("openpbr port parity: subsurface albedo -> sigma inversion") {
	Rng rng = rng_create(31u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		for (u32 c = 0u; c < 3u; ++c) {
			in.a[c] = rng.uniform(0.0f, 1.0f);
			in.b[c] = rng.uniform(0.0f, 2.0f);
		}
		in.a[3] = rng.uniform(-1.0f, 1.0f);
		inputs.emplace_back(in);
	}
	// zero radius hits the mean-free-path floor; g = 1 hits the anisotropy cap
	ParityInput edges = input_zero();
	edges.a[0] = 0.5f;
	edges.a[1] = 1.0f;
	edges.a[3] = 1.0f;
	inputs.emplace_back(edges);
	check_parity(
		"diffusion reflectance",
		ParityFunction::subsurfaceDiffusionReflectance,
		inputs,
		3u
	);
	check_parity(
		"invert diffusion reflectance",
		ParityFunction::subsurfaceInvertDiffusionReflectance,
		inputs,
		3u
	);
	check_parity("sigma absorption", ParityFunction::subsurfaceSigmaAbsorption, inputs, 3u);
	check_parity("sigma scattering", ParityFunction::subsurfaceSigmaScattering, inputs, 3u);
}

TEST_CASE("openpbr port parity: subsurface hg phase and dwivedi guiding") {
	Rng rng = rng_create(32u);
	std::vector<ParityInput> phase;
	std::vector<ParityInput> sampleWo;
	std::vector<ParityInput> dwivedi;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput p = input_zero();
		p.a[0] = rng.uniform(-1.0f, 1.0f);
		// includes the |g| < 1e-3 isotropic branch
		p.a[1] = (i % 8u == 0u) ? 0.0f : rng.uniform(-0.95f, 0.95f);
		phase.emplace_back(p);

		ParityInput w = input_zero();
		rng.unitSphere(w.a);
		w.a[3] = p.a[1];
		w.b[0] = rng.uniform(0.0f, 1.0f);
		w.b[1] = rng.uniform(0.0f, 1.0f);
		sampleWo.emplace_back(w);

		// diffusion length > 1 and its log, as the walk computes them
		ParityInput d = input_zero();
		f32 const albedo = rng.uniform(0.05f, 0.999f);
		d.a[0] = albedo;
		f32 const length = rng.uniform(1.01f, 20.0f);
		d.b[0] = length;
		d.b[1] = std::log((length + 1.0f) / (length - 1.0f));
		d.b[2] = rng.uniform(-1.0f, 1.0f);
		dwivedi.emplace_back(d);
	}
	check_parity("hg eval", ParityFunction::subsurfacePhaseEvalHg, phase, 1u);
	// the sampler reads (u, g) from the same lanes
	for (ParityInput & p : phase) {
		p.a[0] = 0.5f * (p.a[0] + 1.0f);
	}
	check_parity("hg sample cos", ParityFunction::subsurfacePhaseSampleHgCos, phase, 1u);
	check_parity("hg sample wo", ParityFunction::subsurfacePhaseSampleWo, sampleWo, 3u);
	check_parity(
		"dwivedi diffusion length", ParityFunction::dwivediDiffusionLength, dwivedi, 1u
	);
	// eval and sample read (L, phaseLog, x) from a.xyz
	for (ParityInput & d : dwivedi) {
		d.a[0] = d.b[0];
		d.a[1] = d.b[1];
		d.a[2] = d.b[2];
	}
	check_parity("dwivedi phase eval", ParityFunction::dwivediPhaseEval, dwivedi, 1u);
	for (ParityInput & d : dwivedi) {
		d.a[2] = 0.5f * (d.a[2] + 1.0f);
	}
	check_parity(
		"dwivedi phase sample cos", ParityFunction::dwivediPhaseSampleCos, dwivedi, 1u
	);
}

TEST_CASE("openpbr port parity: subsurface entry sampler and pdf") {
	Rng rng = rng_create(33u);
	std::vector<ParityInput> inputs;
	for (u32 i = 0u; i < skFuzzCount; ++i) {
		ParityInput in = input_zero();
		in.a[0] = rng.uniform(0.0f, 1.0f);
		in.a[1] = rng.uniform(0.0f, 1.0f);
		inputs.emplace_back(in);
	}
	check_parity("entry sampler", ParityFunction::subsurfaceEntrySampleWo, inputs, 3u);
	for (ParityInput & in : inputs) {
		in.a[0] = 2.0f * in.a[0] - 1.0f;
	}
	check_parity("entry pdf", ParityFunction::subsurfaceEntryPdf, inputs, 1u);
}
