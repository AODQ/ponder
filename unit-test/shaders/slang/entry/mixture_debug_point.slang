import "util-random";
import "util-material-openpbr";

struct PC {
	u32v4 rootPad[8];
	u64 outVa;
	u32 zeltnerLtcParamHandle;
	f32v3 wi;
	f32 theta0;
	f32 theta1;
	f32 phi0;
	f32 phi1;
	f32 coatWeight;
	f32 coatRoughness;
	f32 coatIor;
	f32 specularWeight;
	f32 specularIor;
	f32 specularRoughness;
	f32 baseMetalness;
	f32 transmissionWeight;
	f32 fuzzWeight;
	f32 fuzzRoughness;
};

[[vk::push_constant]] PC pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
	const f32v3 tanX = f32v3(1.0f, 0.0f, 0.0f);
	const f32v3 tanY = f32v3(0.0f, 1.0f, 0.0f);
	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);
	ShadingFrame frame;
	frame.tanX = tanX; frame.tanY = tanY; frame.nor = nor;
	const ShadingFrame coatFrame = frame;

	MaterialTableHandles tables;
	tables.kullaContyEnergy = 0u;
	tables.zeltnerLtcParam = pc.zeltnerLtcParamHandle;

	OpenPbrMaterial mat;
	mat.coatWeight = pc.coatWeight;
	mat.coatRoughness = pc.coatRoughness;
	mat.coatRoughnessAnisotropy = 0.0f;
	mat.coatIor = pc.coatIor;
	mat.specularWeight = pc.specularWeight;
	mat.specularIor = pc.specularIor;
	mat.specularRoughness = pc.specularRoughness;
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.baseMetalness = pc.baseMetalness;
	mat.transmissionWeight = pc.transmissionWeight;
	mat.transmissionColor = f32v3(1.0f);
	mat.transmissionDepth = 0.0f;
	mat.transmissionDispersionScale = 0.0f;
	mat.transmissionDispersionAbbeNumber = 20.0f;
	// openPbrLobeSelection now reads this to split the diffuse budget; left
	// unset it's uninitialized gpu memory, not a guaranteed 0
	mat.subsurfaceWeight = 0.0f;
	mat.geometryThinWalled = 0.0f;
	mat.fuzzWeight = pc.fuzzWeight;
	mat.fuzzRoughness = pc.fuzzRoughness;

	const OpenPbrLobeSelection sel = openPbrLobeSelection(tables, mat, dot(nor, pc.wi));

	// scan the same 32x32 quadrature sub-grid mixture_expected.comp uses
	// for this bin, tracking the max mixturePdf/transmissionTirPdf/specPdf
	// and the (theta, phi) where the max occurred -- pinpoints a spike a
	// single bin-center probe would miss entirely
	const u32 subGrid = 32u;
	f32 maxMixture = -1.0f;
	f32 maxMixtureTheta = 0.0f;
	f32 maxMixturePhi = 0.0f;
	f32 maxMixtureTirPdf = 0.0f;
	f32 maxMixtureSpecPdf = 0.0f;
	f32 maxMixtureTransPdf = 0.0f;
	f32 maxMixtureDotNorWo = 0.0f;
	f32 maxMixtureDotWiH = 0.0f;
	f32 maxMixtureLenWiPlusWo = 0.0f;
	for (u32 j = 0u; j < subGrid; ++j) {
		const f32 theta = lerp(pc.theta0, pc.theta1, (f32(j) + 0.5f) / f32(subGrid));
		for (u32 k = 0u; k < subGrid; ++k) {
			const f32 phi = lerp(pc.phi0, pc.phi1, (f32(k) + 0.5f) / f32(subGrid));
			const f32 phiFrame = phi - PI;
			const f32 sinTheta = sin(theta);
			const f32v3 wo = f32v3(
				sinTheta * cos(phiFrame), sinTheta * sin(phiFrame), cos(theta)
			);

			const f32 roughenedSpecularRoughness = openPbrCoatRoughenedRoughness(
				mat.specularRoughness, mat.coatRoughness, mat.coatWeight
			);
			const f32v2 specAlpha = openPbrRoughnessAlpha(
				roughenedSpecularRoughness, mat.specularRoughnessAnisotropy
			);
			const ShadingFrame specFrame = shadingFrameRotate(
				frame, mat.specularRoughnessAnisotropyRotation
			);
			const f32 specPdf = utilMicrofacetGgxBoundedReflectPdfAniso(
				specFrame.tanX, specFrame.tanY, specFrame.nor, pc.wi, wo, specAlpha
			);
			const f32 transmissionTirPdf = (
				openPbrTransmissionTirReflectPdf(
					mat, frame, pc.wi, wo, false, mat.specularIor
				)
			);
			const f32 transmissionPdf = (
				openPbrTransmissionPdf(mat, frame, pc.wi, wo, false, mat.specularIor)
			);
			const f32 mixturePdf = openPbrEvaluatePdfSelected(
				sel, mat, frame, coatFrame, pc.wi, wo, false, mat.specularIor
			);

			if (mixturePdf > maxMixture) {
				maxMixture = mixturePdf;
				maxMixtureTheta = theta;
				maxMixturePhi = phi;
				maxMixtureTirPdf = transmissionTirPdf;
				maxMixtureSpecPdf = specPdf;
				maxMixtureTransPdf = transmissionPdf;
				maxMixtureDotNorWo = dot(nor, wo);
				const f32v3 wiPlusWo = pc.wi + wo;
				maxMixtureLenWiPlusWo = length(wiPlusWo);
				const f32v3 h = normalize(wiPlusWo);
				maxMixtureDotWiH = dot(pc.wi, h);
			}
		}
	}

	// fine (256x256, 64x the coarse 32x32 mixture_expected.comp resolution)
	// midpoint quadrature of the same bin, identical integration formula --
	// settles whether a coarse-grid mismatch (quadrature artifact) explains
	// an outlier bin, or whether the analytic pdf genuinely integrates to
	// the same value at any resolution (a real formula/sampler disagreement)
	const u32 fineGrid = 256u;
	f32 fineIntegral = 0.0f;
	for (u32 j = 0u; j < fineGrid; ++j) {
		const f32 theta = lerp(pc.theta0, pc.theta1, (f32(j) + 0.5f) / f32(fineGrid));
		const f32 sinTheta = sin(theta);
		for (u32 k = 0u; k < fineGrid; ++k) {
			const f32 phi = lerp(pc.phi0, pc.phi1, (f32(k) + 0.5f) / f32(fineGrid));
			const f32 phiFrame = phi - PI;
			const f32v3 wo = f32v3(
				sinTheta * cos(phiFrame), sinTheta * sin(phiFrame), cos(theta)
			);
			const f32 pdf = openPbrEvaluatePdfSelected(
				sel, mat, frame, coatFrame, pc.wi, wo, false, mat.specularIor
			);
			fineIntegral += pdf * sinTheta;
		}
	}
	fineIntegral *= (
		(pc.theta1 - pc.theta0) * (pc.phi1 - pc.phi0) / f32(fineGrid * fineGrid)
	);

	// sidedness check at the max-mixture point: reconstruct the same
	// generalized half vector openPbrTransmissionPdf uses and check whether
	// wi and wo actually sit on opposite sides of it (dot(wi,ht) and
	// dot(wo,ht) should have opposite sign for a physically valid refraction
	// configuration -- walter et al. 2007 / pbrt's MicrofacetTransmission
	// both gate on this explicitly; ponder's openPbrTransmissionPdf has no
	// such gate)
	const f32v3 woAtMax = f32v3(
		sin(maxMixtureTheta) * cos(maxMixturePhi - PI),
		sin(maxMixtureTheta) * sin(maxMixturePhi - PI),
		cos(maxMixtureTheta)
	);
	const f32 etaISide = 1.0f;
	const f32 etaTSide = mat.specularIor;
	const f32v3 htUnnormSide = -(etaISide * pc.wi + etaTSide * woAtMax);
	const f32v3 htSide = normalize(htUnnormSide);
	const f32v3 htOrientedSide = dot(htSide, nor) < 0.0f ? -htSide : htSide;
	const f32 dotWiHtSide = dot(pc.wi, htOrientedSide);
	const f32 dotWoHtSide = dot(woAtMax, htOrientedSide);

	f32 * out_ = (f32 *)(pc.outVa);
	out_[0] = maxMixture;
	out_[1] = maxMixtureTheta;
	out_[2] = maxMixturePhi;
	out_[3] = maxMixtureTirPdf;
	out_[4] = maxMixtureSpecPdf;
	out_[5] = maxMixtureDotWiH;
	out_[6] = maxMixtureLenWiPlusWo;
	out_[7] = sel.probabilitySpecular;
	out_[8] = sel.probabilityTransmission;
	out_[9] = sel.probabilityTotal;
	out_[10] = maxMixtureTransPdf;
	out_[11] = maxMixtureDotNorWo;
	out_[12] = fineIntegral;
	out_[13] = dotWiHtSide;
	out_[14] = dotWoHtSide;
}
