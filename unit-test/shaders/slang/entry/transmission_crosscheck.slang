import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-transmission";
#include "../util-fresnel-materialx-reference.slang"
#include "../util-microfacet-materialx-reference.slang"
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// independent cross-check of openPbrTransmissionEvaluateF and
// openPbrTransmissionPdf against a pure-materialx assembly of the walter
// et al. 2007 rough-refraction btdf (eq 21) and its vndf pdf (eq 38/17):
// materialx's own dielectric transmission mode has no direction-pair btdf
// (it only refracts environment lookups), so -- as with the metallic-lobe
// crosscheck -- every per-microfacet factor (D, G1, G2, fresnel) comes
// from the materialx reference ports and only the published walter
// assembly line joins them.
//
// reachable (wi, wo) pairs are generated inside the shader: the vndf
// half-vector sampler draws h from xi, refract() produces wo. the
// reference side then evaluates walter's formula at that exact sampled h,
// while the ponder side reconstructs its own half vector from (wi, wo) --
// so a broken half-vector reconstruction in the lib shows up as a
// mismatch here even though both sides describe the same configuration.
//
// per sample, 8 floats in: wi.xyz, ior, specularRoughness, isInsideMedium
// (0/1), xi.xy. output 8 floats: mineF, refF, minePdf, refPdf, tirFlag,
// wo.z, dotHWi, unused. tint pinned white (transmissionColor = 1,
// depth = 0), thin-walled off, no dispersion.

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 8u;
	const f32v3 wi = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32 ior = inBuf[base+3u];
	const f32 roughness = inBuf[base+4u];
	const bool isInsideMedium = inBuf[base+5u] > 0.5f;
	const f32v2 xi = f32v2(inBuf[base+6u], inBuf[base+7u]);

	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);
	// frisvad basis, the frame the isotropic transmission path used internally
	const ShadingFrame frame = shadingFrameFromNormal(nor);
	const f32 etaI = isInsideMedium ? ior : 1.0f;
	const f32 etaT = isInsideMedium ? 1.0f : ior;

	// production sampling path: vndf half vector, refract
	const f32v3 h = (
		utilMicrofacetSampleGgxVndf(nor, wi, max(roughness, 1e-5f), xi)
	);
	const f32v3 refracted = refract(-wi, h, etaI / etaT);

	f32 * outBuf = (f32 *)(pc.outVa);
	if (length(refracted) < 0.5f) {
		// total internal reflection at this half vector; flagged so the
		// cpu side can skip (the tir fallback path has its own tests)
		outBuf[i*8u+0u] = 0.0f;
		outBuf[i*8u+1u] = 0.0f;
		outBuf[i*8u+2u] = 0.0f;
		outBuf[i*8u+3u] = 0.0f;
		outBuf[i*8u+4u] = 1.0f;
		outBuf[i*8u+5u] = 0.0f;
		outBuf[i*8u+6u] = dot(h, wi);
		outBuf[i*8u+7u] = 0.0f;
		return;
	}
	const f32v3 wo = refracted;

	OpenPbrMaterial mat;
	mat.specularIor = ior;
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.specularRoughness = roughness;
	mat.geometryThinWalled = 0.0f;
	mat.transmissionColor = f32v3(1.0f);
	mat.transmissionDepth = 0.0f;
	mat.transmissionDispersionScale = 0.0f;
	mat.transmissionDispersionAbbeNumber = 0.0f;

	const f32 mineF = (
		openPbrTransmissionEvaluateF(mat, frame, wi, wo, isInsideMedium, ior).x
	);
	const f32 minePdf = (
		openPbrTransmissionPdf(mat, frame, wi, wo, isInsideMedium, ior)
	);

	// materialx-side walter assembly at the sampled half vector itself
	const f32 dotHWi = max(dot(h, wi), 0.0f);
	const f32 dotHWo = abs(dot(h, wo));
	const f32 muI = max(wi.z, 1e-5f);
	const f32 muO = max(abs(wo.z), 1e-5f);
	// length of the unnormalized half-vector sum, not a sum of two
	// independently-clamped cosines -- wi and wo sit on opposite sides of
	// the interface so the signed dot products partially cancel; this is
	// an independent re-derivation (not copied from the lib), cross
	// checked against blender cycles' bsdf_microfacet.h which computes
	// this identically via the raw (pre-normalize) half-vector length
	const f32 denom = length(etaI * wi + etaT * wo);
	const f32 fresnelMx = mxFresnelDielectric(dotHWi, etaT / etaI);
	const f32 dMx = mx_ggx_NDF(h, f32v2(roughness));
	// walter eq 21 with V = G2 / (4 muI muO) folded the same way the lib
	// folds it, so the reference exercises the identical algebraic shape
	// with materialx factors
	const f32 g2Mx = mx_ggx_smith_G2(muO, muI, roughness);
	const f32 vMx = g2Mx / (4.0f * muI * muO);
	const f32 refF = (
		(1.0f - fresnelMx) * dMx * vMx
		* 4.0f * dotHWi * dotHWo * etaT * etaT
		/ (denom * denom)
	);
	// walter eq 38: vndf density of h times the |dwo/dh| refraction
	// jacobian (eq 17)
	const f32 g1Mx = mx_ggx_smith_G1(muI, max(roughness, 1e-5f));
	const f32 dMxFloored = mx_ggx_NDF(h, f32v2(max(roughness, 1e-5f)));
	const f32 refPdf = (
		g1Mx * dMxFloored * dotHWi / muI
		* etaT * etaT * dotHWo
		/ (denom * denom)
	);

	outBuf[i*8u+0u] = mineF;
	outBuf[i*8u+1u] = refF;
	outBuf[i*8u+2u] = minePdf;
	outBuf[i*8u+3u] = refPdf;
	outBuf[i*8u+4u] = 0.0f;
	outBuf[i*8u+5u] = wo.z;
	outBuf[i*8u+6u] = dotHWi;
	outBuf[i*8u+7u] = 0.0f;
}
