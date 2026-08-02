import "util-random";
import "util-material-openpbr";

struct PC {
	u32v4 rootPad[8];
	u64 outVa;
	u32 width;
	u32 height;
	u32 excludedLobe;
	u32 zeltnerLtcParamHandle;
	f32 phiWo;
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

// dense (wi grazing x full-sphere wo) sweep of openPbrEvaluatePdfSelected
// for a material with every lobe active except pc.excludedLobe, which is
// pinned to exactly zero weight/probability. also evaluates that excluded
// lobe's own pdf primitive directly and unguarded at the same (wi, wo), so
// the c++ side can confirm the zero-probability guard
// (util-material-openpbr.glsl's `sel.probabilityX > 0.0f ? ... : 0.0f`
// terms) is doing real work -- excluding a term that would otherwise have
// been non-finite -- rather than the test being vacuous because the
// excluded lobe's raw pdf happened to be finite anyway everywhere swept.
// full [0,pi] theta range on wo (not just above-horizon) so this also
// exercises openPbrEvaluatePdfSelected's dotNorWo<=0 leak branch, where up
// to three of {transmission, specular-leak, coat-leak} are simultaneously
// guarded and summed.

// per texel, 4 floats out: mixturePdf, rawExcludedPdf, mixtureFinite (1/0),
// rawExcludedFinite (1/0)

func is_finite(
	v: f32,
) -> bool {
	return !isnan(v) && !isinf(v);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 x = gl_GlobalInvocationID.x;
	const u32 y = gl_GlobalInvocationID.y;
	if (x >= pc.width || y >= pc.height) { return; }

	const f32v3 tanX = f32v3(1.0f, 0.0f, 0.0f);
	const f32v3 tanY = f32v3(0.0f, 1.0f, 0.0f);
	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);
	ShadingFrame frame;
	frame.tanX = tanX;
	frame.tanY = tanY;
	frame.nor = nor;
	const ShadingFrame coatFrame = frame;

	// x: cosThetaWi in (0, 1], grazing near x=0, matching the ggx/
	// transmission sweep convention (production wi is always above horizon)
	const f32 cosThetaWi = max((f32(x) + 0.5f) / f32(pc.width), 1e-5f);
	const f32 sinThetaWi = sqrt(max(0.0f, 1.0f - cosThetaWi * cosThetaWi));
	const f32v3 wi = f32v3(sinThetaWi, 0.0f, cosThetaWi);

	// y: thetaWo over the full [0, pi] range, so both the ordinary
	// above-horizon pdf and the dotNorWo<=0 leak branch get swept
	const f32 thetaWo = PI * (f32(y) + 0.5f) / f32(pc.height);
	const f32 sinThetaWo = sin(thetaWo);
	const f32 cosThetaWo = cos(thetaWo);
	const f32v3 wo = (
		f32v3(
			sinThetaWo * cos(pc.phiWo),
			sinThetaWo * sin(pc.phiWo),
			cosThetaWo
		)
	);

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

	// pin the excluded lobe's own weight to exactly zero, so
	// sel.probabilityX == 0.0f for it and the mixture guard is genuinely
	// exercised (not just numerically tiny)
	if (pc.excludedLobe == OPENPBR_LOBE_COAT) { mat.coatWeight = 0.0f; }
	else if (pc.excludedLobe == OPENPBR_LOBE_SPECULAR) {
		mat.specularWeight = 0.0f;
		mat.baseMetalness = 0.0f;
	} else if (pc.excludedLobe == OPENPBR_LOBE_FUZZ) { mat.fuzzWeight = 0.0f; }
	else if (pc.excludedLobe == OPENPBR_LOBE_TRANSMISSION) {
		mat.transmissionWeight = 0.0f;
	}
	// diffuse's probability can only be pinned to zero by routing its
	// entire budget elsewhere (metalness=1 or transmission=1); routing all
	// of it to transmission keeps the leak branch's transmission term live
	if (pc.excludedLobe == OPENPBR_LOBE_DIFFUSE) {
		mat.transmissionWeight = 1.0f;
	}

	const OpenPbrLobeSelection sel = (
		openPbrLobeSelection(tables, mat, dot(nor, wi))
	);

	const f32 mixturePdf = (
		openPbrEvaluatePdfSelected(
			sel, mat, frame, coatFrame, wi, wo,
			/*isInsideMedium=*/false, /*dispersedIor=*/mat.specularIor
		)
	);

	// the excluded lobe's own pdf primitive, called directly and
	// unguarded, at the identical (wi, wo) -- this is what the guard is
	// standing between and the mixture sum
	f32 rawExcludedPdf;
	if (pc.excludedLobe == OPENPBR_LOBE_COAT) {
		const f32v2 coatAlpha = (
			openPbrRoughnessAlpha(mat.coatRoughness, mat.coatRoughnessAnisotropy)
		);
		rawExcludedPdf = (
			utilMicrofacetGgxBoundedReflectPdfAniso(
				coatFrame.tanX, coatFrame.tanY, coatFrame.nor, wi, wo, coatAlpha
			)
		);
	} else if (pc.excludedLobe == OPENPBR_LOBE_SPECULAR) {
		const f32 roughenedSpecRoughness = (
			openPbrCoatRoughenedRoughness(
				mat.specularRoughness, mat.coatRoughness, mat.coatWeight
			)
		);
		const f32v2 specAlpha = (
			openPbrRoughnessAlpha(
				roughenedSpecRoughness, mat.specularRoughnessAnisotropy
			)
		);
		const ShadingFrame specFrame = (
			shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
		);
		rawExcludedPdf = (
			utilMicrofacetGgxBoundedReflectPdfAniso(
				specFrame.tanX, specFrame.tanY, specFrame.nor, wi, wo, specAlpha
			)
		);
	} else if (pc.excludedLobe == OPENPBR_LOBE_DIFFUSE) {
		rawExcludedPdf = openPbrGlossyDiffusePdf(dot(nor, wo));
	} else if (pc.excludedLobe == OPENPBR_LOBE_FUZZ) {
		rawExcludedPdf = openPbrFuzzPdf(nor, wi, wo, sel.fuzzA, sel.fuzzB);
	} else {
		rawExcludedPdf = (
			openPbrTransmissionPdf(mat, frame, wi, wo, false, mat.specularIor)
			+ openPbrTransmissionTirReflectPdf(
				mat, frame, wi, wo, false, mat.specularIor
			)
		);
	}

	f32 * out_ = (f32 *)(pc.outVa);
	const u32 base = (y * pc.width + x) * 4u;
	out_[base+0u] = mixturePdf;
	out_[base+1u] = rawExcludedPdf;
	out_[base+2u] = is_finite(mixturePdf) ? 1.0f : 0.0f;
	out_[base+3u] = is_finite(rawExcludedPdf) ? 1.0f : 0.0f;
}
