import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-diffuse";
import "util-material-openpbr-fuzz";
import "util-material-openpbr-transmission";
#include "../util-sampling-bins.slang"
#include "../util-microfacet-materialx-reference.slang"
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 expectedVa;
	u32 mode;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 wi;
	f32v2 alpha;
	f32v2 aniso;
};

[[vk::push_constant]] PC pc;

// expected side of the sampler-vs-pdf chi-square comparison (modeled on
// cull app/shaders/bsdf-verify-expected.comp): one thread per bin,
// numerically integrates the mode's pdf over the bin's solid angle and
// scales by the sample count.
//
// the vndf mode's expected density is assembled purely from the materialx
// reference functions (mx_ggx_NDF, mx_ggx_smith_G1) rather than any ponder
// code, so for that mode the chi-square doubles as an independent
// cross-check of utilMicrofacetSampleGgxVndf against materialx.

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

#define SAMPLING_MODE_GGX_BOUNDED_WO 0u
#define SAMPLING_MODE_GGX_VNDF_H 1u
#define SAMPLING_MODE_DIFFUSE_WO 2u
#define SAMPLING_MODE_FUZZ_WO 3u
#define SAMPLING_MODE_TRANSMISSION_ENTER 4u
#define SAMPLING_MODE_TRANSMISSION_EXIT 5u

// same canonical frame and push layout as sampling_histogram.comp

func pdf_for_mode(
	wo: f32v3,
) -> f32 {
	const f32v3 tanX = f32v3(1.0f, 0.0f, 0.0f);
	const f32v3 tanY = f32v3(0.0f, 1.0f, 0.0f);
	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);
	// the canonical frame above, matching sampling_histogram.comp's sampler
	ShadingFrame frame;
	frame.tanX = tanX;
	frame.tanY = tanY;
	frame.nor = nor;
	if (pc.mode == SAMPLING_MODE_GGX_BOUNDED_WO) {
		return (
			utilMicrofacetGgxBoundedReflectPdfAniso(
				tanX, tanY, nor, pc.wi, wo, pc.alpha
			)
		);
	}
	if (pc.mode == SAMPLING_MODE_GGX_VNDF_H) {
		// vndf density of the microfacet normal h (heitz 2018 eq 3),
		// assembled entirely from materialx pieces:
		// D_vis(h) = G1(wi) * D(h) * max(0, wi.h) / wi.z
		if (wo.z <= 0.0f) {
			return 0.0f;
		}
		const f32 g1 = mx_ggx_smith_G1(pc.wi.z, pc.alpha.x);
		const f32 d = mx_ggx_NDF(wo, f32v2(pc.alpha.x));
		return g1 * d * max(dot(pc.wi, wo), 0.0f) / pc.wi.z;
	}
	if (
		pc.mode == SAMPLING_MODE_TRANSMISSION_ENTER
		|| pc.mode == SAMPLING_MODE_TRANSMISSION_EXIT
	) {
		// density of the production transmission draw path: the walter
		// refraction pdf on the transmitted side plus the vndf reflect
		// density restricted to the actually-TIR half-vector region on the
		// reflected side. the transmitted side is still restricted here:
		// the raw openPbrTransmissionPdf formula also returns nonzero
		// density at wrong-branch (mirror-side) directions the sampler can
		// never emit, same harness-side domain-restriction convention as
		// the max(pdf, 0) clamps for diffuse/fuzz below. the reflected
		// side's restriction below is now redundant with
		// openPbrTransmissionTirReflectPdf's own internal TIR check (fixed
		// after the mixture-lobe-selection milestone found the previously-
		// unrestricted version double-counting against a separately-
		// normalized reflecting lobe in the full mixture pdf) -- left in
		// place since it's harmless and this file predates that fix
		const bool isInsideMedium = pc.mode == SAMPLING_MODE_TRANSMISSION_EXIT;
		const f32 ior = pc.alpha.y;
		OpenPbrMaterial mat;
		mat.specularIor = ior;
		mat.specularRoughnessAnisotropy = pc.aniso.x;
		mat.specularRoughnessAnisotropyRotation = pc.aniso.y;
		mat.specularRoughness = pc.alpha.x;
		mat.geometryThinWalled = 0.0f;
		mat.transmissionColor = f32v3(1.0f);
		mat.transmissionDepth = 0.0f;
		mat.transmissionDispersionScale = 0.0f;
		mat.transmissionDispersionAbbeNumber = 0.0f;
		const f32 etaI = isInsideMedium ? ior : 1.0f;
		const f32 etaT = isInsideMedium ? 1.0f : ior;
		f32 pdf = 0.0f;
		// transmitted side: exact refract-image support test (reconstruct
		// the half vector the lib way, refract about it, require it to
		// land back on wo)
		const f32v3 ht = normalize(-(etaI * pc.wi + etaT * wo));
		const f32v3 htOriented = dot(ht, nor) < 0.0f ? -ht : ht;
		const f32v3 refr = refract(-pc.wi, htOriented, etaI / etaT);
		if (
			length(refr) > 0.5f
			&& dot(normalize(refr), wo) > 1.0f - 1e-5f
		) {
			pdf += (
				openPbrTransmissionPdf(
					mat, frame, pc.wi, wo, isInsideMedium, ior
				)
			);
		}
		// reflected side: only half vectors that actually undergo total
		// internal reflection (glsl refract's k < 0) are reachable
		const f32v3 h = normalize(pc.wi + wo);
		const f32 dotWiH = dot(pc.wi, h);
		const f32 eta = etaI / etaT;
		const f32 k = 1.0f - eta * eta * (1.0f - dotWiH * dotWiH);
		if (dot(nor, h) > 0.0f && dotWiH > 0.0f && k < 0.0f) {
			pdf += (
				openPbrTransmissionTirReflectPdf(
					mat, frame, pc.wi, wo, isInsideMedium, ior
				)
			);
		}
		return pdf;
	}
	if (pc.mode == SAMPLING_MODE_DIFFUSE_WO) {
		// diffuse: openPbrGlossyDiffusePdf is only ever evaluated at
		// sampled (above-horizon) directions in production; the sampler
		// never emits below the horizon, so the expected density there is
		// zero rather than the (negative) raw cosine the formula would
		// return
		return max(openPbrGlossyDiffusePdf(wo.z), 0.0f);
	}
	// fuzz: same convention -- the sampler only ever emits into the
	// nor-upper hemisphere (see the fuzz sampler doc comment), and
	// openPbrFuzzPdf has no clamp of its own
	return max(openPbrFuzzPdf(nor, pc.wi, wo, pc.alpha.x, pc.alpha.y), 0.0f);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 bin = gl_GlobalInvocationID.x;
	if (bin >= BSDF_VERIFY_BIN_COUNT) {
		return;
	}
	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);

	// this bin's (theta, phi) rectangle
	const u32 thetaBin = bin / BSDF_VERIFY_PHI_BINS;
	const u32 phiBin = bin % BSDF_VERIFY_PHI_BINS;
	const f32 theta0 = PI * f32(thetaBin) / f32(BSDF_VERIFY_THETA_BINS);
	const f32 theta1 = PI * f32(thetaBin + 1u) / f32(BSDF_VERIFY_THETA_BINS);
	const f32 phi0 = TAU * f32(phiBin) / f32(BSDF_VERIFY_PHI_BINS);
	const f32 phi1 = TAU * f32(phiBin + 1u) / f32(BSDF_VERIFY_PHI_BINS);

	// midpoint over the subgrid.
	// pdf(\omega) * sin(\theta), d_{omega} = sin(\theta) d\theta d\phi
	// integration error should stay under sqrt(E) per bin. 32x32 rather
	// than cull's 8x8: the bounded sampler's support cut is a curve
	// through bin interiors, and at 8x8 the partial-coverage bins along
	// it bias E low by far more than sqrt(E) (measured: pooled-bin terms
	// of 67^2 and 250^2 at grazing incidence)
	const u32 subGrid = 32u;
	f32 integral = 0.0f;
	for (u32 j = 0u; j < subGrid; ++j) {
		const f32 theta = lerp(theta0, theta1, (f32(j) + 0.5f) / f32(subGrid));
		const f32 sinTheta = sin(theta);
		for (u32 k = 0u; k < subGrid; ++k) {
			const f32 phi = lerp(phi0, phi1, (f32(k) + 0.5f) / f32(subGrid));
			// same frame + phi convention as the histogram
			const f32v3 wo = (
				bsdfVerifyDirectionFromAngles(nor, theta, phi)
			);
			integral += pdf_for_mode(wo) * sinTheta;
		}
	}
	integral *= (
		(theta1 - theta0) * (phi1 - phi0) / f32(subGrid * subGrid)
	);

	((f32 *)(pc.expectedVa))[bin] = integral * f32(pc.sampleCount);
}
