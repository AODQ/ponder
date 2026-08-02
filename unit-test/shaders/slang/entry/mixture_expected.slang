import "util-random";
import "util-material-openpbr";
#include "../util-sampling-bins.slang"

struct PC {
	u32v4 rootPad[8];
	u64 expectedVa;
	u32 zeltnerLtcParamHandle;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 wi;
	f32 coatWeight;
	f32 coatRoughness;
	f32 coatRoughnessAnisotropy;
	f32 coatIor;
	f32 specularWeight;
	f32 specularIor;
	f32 specularRoughness;
	f32 specularRoughnessAnisotropy;
	f32 specularRoughnessAnisotropyRotation;
	f32 baseMetalness;
	f32 transmissionWeight;
	f32 geometryThinWalled;
	f32 fuzzWeight;
	f32 fuzzRoughness;
};

[[vk::push_constant]] PC pc;

// expected side of the multi-lobe mixture chi-square: one thread per bin,
// numerically integrates openPbrEvaluatePdfSelected (the real production
// mixture pdf, weighting each lobe's pdf by the identical sel.probabilityX
// fractions the sampler used) over the bin's solid angle. same quadrature
// convention as sampling_expected.comp. see mixture_histogram.comp for why
// this pulls in util-material-openpbr.glsl wholesale rather than the
// per-lobe files sampling_expected.comp uses.

// same push layout as mixture_histogram.comp, expectedVa in place of
// histogramVa/pickCountVa

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 bin = gl_GlobalInvocationID.x;
	if (bin >= BSDF_VERIFY_BIN_COUNT) {
		return;
	}
	const f32v3 tanX = f32v3(1.0f, 0.0f, 0.0f);
	const f32v3 tanY = f32v3(0.0f, 1.0f, 0.0f);
	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);
	ShadingFrame frame;
	frame.tanX = tanX;
	frame.tanY = tanY;
	frame.nor = nor;
	const ShadingFrame coatFrame = frame;

	MaterialTableHandles tables;
	tables.kullaContyEnergy = 0u;
	tables.zeltnerLtcParam = pc.zeltnerLtcParamHandle;

	OpenPbrMaterial mat;
	mat.coatWeight = pc.coatWeight;
	mat.coatRoughness = pc.coatRoughness;
	mat.coatRoughnessAnisotropy = pc.coatRoughnessAnisotropy;
	mat.coatIor = pc.coatIor;
	mat.specularWeight = pc.specularWeight;
	mat.specularIor = pc.specularIor;
	mat.specularRoughness = pc.specularRoughness;
	mat.specularRoughnessAnisotropy = pc.specularRoughnessAnisotropy;
	mat.specularRoughnessAnisotropyRotation = (
		pc.specularRoughnessAnisotropyRotation
	);
	mat.baseMetalness = pc.baseMetalness;
	mat.transmissionWeight = pc.transmissionWeight;
	mat.transmissionColor = f32v3(1.0f);
	mat.transmissionDepth = 0.0f;
	// openPbrLobeSelection now reads this to split the diffuse budget; left
	// unset it's uninitialized gpu memory, not a guaranteed 0
	mat.subsurfaceWeight = 0.0f;
	mat.transmissionDispersionScale = 0.0f;
	mat.transmissionDispersionAbbeNumber = 20.0f;
	mat.geometryThinWalled = pc.geometryThinWalled;
	mat.fuzzWeight = pc.fuzzWeight;
	mat.fuzzRoughness = pc.fuzzRoughness;

	const OpenPbrLobeSelection sel = (
		openPbrLobeSelection(tables, mat, dot(nor, pc.wi))
	);

	// this bin's (theta, phi) rectangle
	const u32 thetaBin = bin / BSDF_VERIFY_PHI_BINS;
	const u32 phiBin = bin % BSDF_VERIFY_PHI_BINS;
	const f32 theta0 = PI * f32(thetaBin) / f32(BSDF_VERIFY_THETA_BINS);
	const f32 theta1 = PI * f32(thetaBin + 1u) / f32(BSDF_VERIFY_THETA_BINS);
	const f32 phi0 = TAU * f32(phiBin) / f32(BSDF_VERIFY_PHI_BINS);
	const f32 phi1 = TAU * f32(phiBin + 1u) / f32(BSDF_VERIFY_PHI_BINS);

	// same 32x32 midpoint quadrature as sampling_expected.comp
	const u32 subGrid = 32u;
	f32 integral = 0.0f;
	for (u32 j = 0u; j < subGrid; ++j) {
		const f32 theta = lerp(theta0, theta1, (f32(j) + 0.5f) / f32(subGrid));
		const f32 sinTheta = sin(theta);
		for (u32 k = 0u; k < subGrid; ++k) {
			const f32 phi = lerp(phi0, phi1, (f32(k) + 0.5f) / f32(subGrid));
			const f32v3 wo = bsdfVerifyDirectionFromAngles(nor, theta, phi);
			const f32 pdf = (
				openPbrEvaluatePdfSelected(
					sel, mat, frame, coatFrame, pc.wi, wo,
					/*isInsideMedium=*/false, /*dispersedIor=*/mat.specularIor
				)
			);
			integral += pdf * sinTheta;
		}
	}
	integral *= (
		(theta1 - theta0) * (phi1 - phi0)
		/ f32(subGrid * subGrid)
	);

	const f32 scaled = integral * f32(pc.sampleCount);

	if (bin == 0u) {
		printf(
			"mixture_expected bin0 write integral=%f scaled=%f",
			integral, scaled
		);
	}

	((f32 *)(pc.expectedVa))[bin] = scaled;
}
