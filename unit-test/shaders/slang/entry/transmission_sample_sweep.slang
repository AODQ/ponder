import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-transmission";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 outVa;
	u32 width;
	u32 height;
	f32 ior;
	f32 isInsideMedium;
};

[[vk::push_constant]] PC pc;

// dense parameter-space sweep hunting for NaN/Inf and degenerate output in
// the production transmission sampling path (vndf half vector -> refract,
// with the total-internal-reflection reflect fallback) and the pdfs that
// own each branch. one thread per (incidence x roughness) texel, ior and
// medium side fixed per dispatch; each texel exercises an 8x8 xi grid
// including the exact 0 endpoint (the ~1 upper stratum is carved into its
// own tests, matching the ggx/fuzz sweep convention).

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// texel (x, y) maps to cosThetaI in [-1, 1] over x (below-horizon wi
// included to document behavior outside the production domain) and
// specularRoughness in [1e-5, 1] over y. output 4 floats per texel:
// nonFiniteCount (wo or own-branch pdf), maxUnitLengthError,
// minOwnBranchPdf, tirFraction.

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32v2 texel = gl_GlobalInvocationID.xy;
	if (texel.x >= pc.width || texel.y >= pc.height) { return; }

	const f32 cosThetaI = (
		-1.0f + 2.0f * f32(texel.x) / f32(pc.width - 1u)
	);
	const f32 sinThetaI = sqrt(max(0.0f, 1.0f - cosThetaI * cosThetaI));
	const f32v3 wi = f32v3(sinThetaI, 0.0f, cosThetaI);
	const f32 roughness = (
		1e-5f + (1.0f - 1e-5f) * f32(texel.y) / f32(pc.height - 1u)
	);
	const bool isInsideMedium = pc.isInsideMedium > 0.5f;

	OpenPbrMaterial mat;
	mat.specularIor = pc.ior;
	// identities for the coat/weight ior modulation openPbrTransmissionIor
	// now applies, so etaI/etaT below stay pc.ior exactly
	mat.specularWeight = 1.0f;
	mat.coatWeight = 0.0f;
	mat.coatIor = 1.0f;
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.specularRoughness = roughness;
	mat.geometryThinWalled = 0.0f;
	mat.transmissionColor = f32v3(1.0f);
	mat.transmissionDepth = 0.0f;
	mat.transmissionDispersionScale = 0.0f;
	mat.transmissionDispersionAbbeNumber = 0.0f;

	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);
	// frisvad basis, the frame the isotropic transmission path used internally
	const ShadingFrame frame = shadingFrameFromNormal(nor);
	const f32 etaI = isInsideMedium ? pc.ior : 1.0f;
	const f32 etaT = isInsideMedium ? 1.0f : pc.ior;

	f32 nonFiniteCount = 0.0f;
	f32 maxLenErr = 0.0f;
	f32 minPdf = 1e30f;
	f32 tirCount = 0.0f;

	const u32 kXiSteps = 8u;
	for (u32 xj = 0u; xj < kXiSteps; ++xj) {
		for (u32 xk = 0u; xk < kXiSteps; ++xk) {
			const f32v2 xi = f32v2(
				f32(xj) / f32(kXiSteps), f32(xk) / f32(kXiSteps)
			);
			// the production transmission draw path from openPbrSampleWo
			const f32v3 h = (
				utilMicrofacetSampleGgxVndf(
					nor, wi, max(roughness, 1e-5f), xi
				)
			);
			const f32v3 refracted = refract(-wi, h, etaI / etaT);
			f32v3 wo;
			f32 pdf;
			if (length(refracted) > 0.5f) {
				wo = refracted;
				pdf = (
					openPbrTransmissionPdf(
						mat, frame, wi, wo, isInsideMedium, pc.ior
					)
				);
			} else {
				tirCount += 1.0f;
				wo = normalize(2.0f * dot(wi, h) * h - wi);
				pdf = (
					openPbrTransmissionTirReflectPdf(
						mat, frame, wi, wo, isInsideMedium, pc.ior
					)
				);
			}

			const bool finite = (
				!isnan(wo.x) && !isnan(wo.y) && !isnan(wo.z)
				&& !isinf(wo.x) && !isinf(wo.y) && !isinf(wo.z)
				&& !isnan(pdf) && !isinf(pdf)
			);
			if (!finite) {
				nonFiniteCount += 1.0f;
				continue;
			}
			maxLenErr = max(maxLenErr, abs(length(wo) - 1.0f));
			minPdf = min(minPdf, pdf);
		}
	}

	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 base = (texel.y * pc.width + texel.x) * 4u;
	outBuf[base+0u] = nonFiniteCount;
	outBuf[base+1u] = maxLenErr;
	outBuf[base+2u] = minPdf;
	outBuf[base+3u] = tirCount / f32(kXiSteps * kXiSteps);
}
