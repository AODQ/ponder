import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-diffuse";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 thetaSteps;
	u32 phiSteps;
	u32 count;
};

[[vk::push_constant]] PC pc;

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the diffuse-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// identical to openPbrGlossyDiffuseLayerEvaluateF in
// glossy_diffuse_layer_evaluate.comp; duplicated rather than shared
// because glslangValidator has no notion of a shared test-only header
// beyond #include, and the two dispatches already diverge on push-constant
// layout. keep both in sync with openPbrEvaluateF's fSpecular/fDielectricBase
// block if that ever changes.
func openPbrGlossyDiffuseLayerEvaluateF(
	mat: OpenPbrMaterial,
	frame: ShadingFrame,
	wi: f32v3,
	wo: f32v3,
) -> f32v3 {
	const f32v3 nor = frame.nor;

	const f32 etaEff = openPbrEffectiveIor(mat.specularIor, mat.specularWeight);
	const f32 etaRel = etaEff;
	const f32 iorRatio = (etaEff - 1.0f) / (etaEff + 1.0f);
	const f32 f0Eff = iorRatio * iorRatio;

	const f32v3 fDiffuse = (
		openPbrGlossyDiffuseOrenNayerEvaluateF(mat, nor, wi, wo)
	);

	const f32v3 fSpecular = (
		openPbrDielectricSpecularEvaluateF(mat, frame, wi, wo, etaRel)
		* mat.specularColor
	);
	const f32 roughenedSpecularRoughness = (
		openPbrCoatRoughenedRoughness(
			mat.specularRoughness, mat.coatRoughness, mat.coatWeight
		)
	);
	const f32 fSpecularDirectionalAlbedo = (
		utilMicrofacetDielectricAlbedo(
			dot(nor, wi), roughenedSpecularRoughness, f0Eff
		)
	);

	return fSpecular + ((1.0f - fSpecularDirectionalAlbedo) * fDiffuse);
}

// furnace test: numerically integrates the combined glossy-diffuse layer
// over the wo hemisphere for a FIXED wi, under constant unit illumination
// -- the same roles the renderer uses (wi is the camera-side direction,
// wo is what gets integrated/sampled). openpbr's albedo-scaling evaluates
// Rs at that fixed camera-side direction (spec's \omega_o) precisely so it
// pulls out of the integral:
//   integral_wo [fSpecular(wi,wo) + (1-Rs(wi))*fDiffuse(wi,wo)] dwo
//     = R_spec_actual(wi) + (1-Rs(wi)) * R_diffuse_actual(wi)
// for a fully white, fully-weighted material this is
// R_spec_actual(wi) + (1-Rs(wi))*1 (diffuse integrates to 1, proven by the
// eon furnace test), which is exactly 1 to the extent Rs(wi) approximates
// R_spec_actual(wi) -- the SAME turquin-compensation approximation error
// already bounded generously in test-dielectric-specular.cpp's furnace
// test, not a new source of error introduced by this layer.
//
// per sample, 8 floats: muO (fixed view cosine), specularRoughness,
// specularRoughnessAnisotropy, specularIor, baseDiffuseRoughness (sigma),
// baseColorScalar, specularColorScalar, baseWeight (specularWeight fixed
// to 1; see note in the golden-value test for why weight=0 fully removes
// the lobe already)

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 8u;
	const f32 muO = inBuf[base+0u];

	OpenPbrMaterial mat;
	mat.specularRoughness = inBuf[base+1u];
	mat.specularRoughnessAnisotropy = inBuf[base+2u];
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.specularIor = inBuf[base+3u];
	mat.specularWeight = 1.0f;
	mat.baseDiffuseRoughness = inBuf[base+4u];
	mat.baseColor = f32v3(inBuf[base+5u]);
	mat.specularColor = f32v3(inBuf[base+6u]);
	mat.baseWeight = inBuf[base+7u];
	mat.coatWeight = 0.0f;
	mat.coatRoughness = 0.0f;
	mat.thinFilmWeight = 0.0f;

	ShadingFrame frame;
	frame.tanX = f32v3(1.0f, 0.0f, 0.0f);
	frame.tanY = f32v3(0.0f, 1.0f, 0.0f);
	frame.nor = f32v3(0.0f, 0.0f, 1.0f);

	// fixed camera-side direction -- the renderer passes this as wi
	const f32v3 wi = f32v3(sqrt(max(0.0f, 1.0f - muO * muO)), 0.0f, muO);

	const f32 dTheta = (PI * 0.5f) / f32(pc.thetaSteps);
	const f32 dPhi = (2.0f * PI) / f32(pc.phiSteps);

	f32 sum = 0.0f;
	for (u32 ti = 0u; ti < pc.thetaSteps; ++ti) {
		const f32 theta = (f32(ti) + 0.5f) * dTheta;
		const f32 cosTheta = cos(theta);
		const f32 sinTheta = sin(theta);
		for (u32 pi = 0u; pi < pc.phiSteps; ++pi) {
			const f32 phi = (f32(pi) + 0.5f) * dPhi;
			const f32v3 wo = (
				f32v3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta)
			);
			const f32v3 f = (
				openPbrGlossyDiffuseLayerEvaluateF(mat, frame, wi, wo)
			);
			sum += f.x * cosTheta * sinTheta * dTheta * dPhi;
		}
	}
	((f32 *)(pc.outVa))[i] = sum;
}
