import "util-shading-frame";
import "util-fresnel";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// per sample, 12 floats: wi.xyz, wo.xyz, specularRoughness,
// specularRoughnessAnisotropy, specularRoughnessAnisotropyRotation,
// coatRoughness, coatWeight, etaRel (coatRoughness/coatWeight/etaRel
// unused by the reciprocity test but kept for layout symmetry with the
// full-evaluate shader)

// test-only: same as openPbrDielectricSpecularEvaluateF but without the
// (view-only-dependent, reciprocity-breaking) turquin energy compensation
// multiplier -- used to test reciprocity of just the base fresnel*D*V lobe
func evaluateUncompensated(
	specularRoughness: f32,
	specularRoughnessAnisotropy: f32,
	specularRoughnessAnisotropyRotation: f32,
	coatRoughness: f32,
	coatWeight: f32,
	frame: ShadingFrame,
	wi: f32v3,
	wo: f32v3,
	etaRel: f32,
) -> f32v3 {
	const f32 roughenedRoughness = (
		openPbrCoatRoughenedRoughness(specularRoughness, coatRoughness, coatWeight)
	);
	const f32v2 alpha = (
		openPbrRoughnessAlpha(roughenedRoughness, specularRoughnessAnisotropy)
	);
	const ShadingFrame specFrame = (
		shadingFrameRotate(frame, specularRoughnessAnisotropyRotation)
	);
	const f32v3 wiLocal = (
		f32v3(
			dot(wi, specFrame.tanX), dot(wi, specFrame.tanY), dot(wi, specFrame.nor)
		)
	);
	const f32v3 woLocal = (
		f32v3(
			dot(wo, specFrame.tanX), dot(wo, specFrame.tanY), dot(wo, specFrame.nor)
		)
	);
	const f32v3 hLocal = normalize(wiLocal + woLocal);
	const f32 dotHWi = dot(hLocal, wiLocal);
	const f32v3 mfFresnel = f32v3(utilFresnelDielectric(dotHWi, etaRel));
	const f32 mfDistribution = utilMicrofacetGgxDistributionAniso(hLocal, alpha);
	const f32 mfVisibility = (
		utilMicrofacetSmithGgxVisibilityAniso(wiLocal, woLocal, alpha)
	);
	return mfFresnel * mfDistribution * mfVisibility;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 12u;
	const f32v3 wi = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32v3 wo = (
		f32v3(inBuf[base+3u], inBuf[base+4u], inBuf[base+5u])
	);
	const f32 specularRoughness = inBuf[base+6u];
	const f32 specularRoughnessAnisotropy = inBuf[base+7u];
	const f32 specularRoughnessAnisotropyRotation = inBuf[base+8u];
	const f32 coatRoughness = inBuf[base+9u];
	const f32 coatWeight = inBuf[base+10u];
	const f32 etaRel = inBuf[base+11u];

	ShadingFrame frame;
	frame.tanX = f32v3(1.0f, 0.0f, 0.0f);
	frame.tanY = f32v3(0.0f, 1.0f, 0.0f);
	frame.nor = f32v3(0.0f, 0.0f, 1.0f);

	((f32v3 *)(pc.outVa))[i] = evaluateUncompensated(
		specularRoughness, specularRoughnessAnisotropy,
		specularRoughnessAnisotropyRotation, coatRoughness, coatWeight,
		frame, wi, wo, etaRel
	);
}
