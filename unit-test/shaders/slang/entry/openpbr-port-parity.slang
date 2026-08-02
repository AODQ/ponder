// parity of the new-core openpbr port (lib/ponder/shaders, module bsdf)
// against the old pipeline (shaders-old) and, where one exists, materialx.
// compiled with PONDER_SPECTRUM_RGB so spectral values compare per channel.

// -- old pipeline
import "util-fresnel";
import "util-material-openpbr-microfacet";
import "util-roughness";
import "util-energy-compensation";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-metallic";
import "util-material-openpbr-diffuse";
import "util-material-openpbr-coat";
import "util-material-openpbr-fuzz";
import "util-material-openpbr-transmission";
import "util-material-openpbr-subsurface";
#include "../util-fresnel-materialx-reference.slang"
#include "../util-microfacet-materialx-reference.slang"
#include "../util-eon-materialx-reference.slang"

// -- new core
import bsdf;
import material;
import util;

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
	u32 function;
	u32 zeltnerLtcParamHandle;
};

[[vk::push_constant]] PC pc;

// per sample: 9 f32v4 of inputs (a .. i)
// per sample: 3 f32v4 of outputs (new, old, materialx); unused lanes are 0
enum ParityFunction : u32 {
	FresnelDielectric = 0,
	FresnelMetallic = 1,
	GgxDistributionAniso = 2,
	SmithVisibilityAniso = 3,
	SmithG1Aniso = 4,
	SampleGgxBoundedWoAniso = 5,
	SampleGgxVndfAniso = 6,
	GgxBoundedReflectPdfAniso = 7,
	GgxDirectionalAlbedo = 8,
	DielectricEnergyCompensate = 9,
	DielectricAlbedo = 10,
	RoughnessAlpha = 11,
	CoatRoughenedRoughness = 12,
	CoatAffectedIor = 13,
	EffectiveIor = 14,
	SpecularEffectiveIor = 15,
	IorFromF0 = 16,
	HalfVectorLocal = 17,
	FresnelDielectricPhase = 18,
	FresnelConductorPhase = 19,
	EvaluateXyzSensitivity = 20,
	ThinfilmFresnel = 21,
	DielectricSpecularEvaluateF = 22,
	FresnelMetallicEvaluateF = 23,
	OrenNayerEnergyCompensate = 24,
	GlossyDiffuseEvaluateF = 25,
	ThinWalledSubsurfaceEvaluateF = 26,
	ThinWalledSubsurfacePdf = 27,
	GlossyDiffusePdf = 28,
	ZeltnerFuzzLookup = 29,
	FuzzEvaluateF = 30,
	FuzzPdf = 31,
	FuzzSampleWo = 32,
	CoatEvaluateF = 33,
	TransmissionEvaluateF = 34,
	TransmissionPdf = 35,
	TransmissionTirReflectPdf = 36,
	DispersionIorRgb = 37,
	SubsurfaceDiffusionReflectance = 38,
	SubsurfaceInvertDiffusionReflectance = 39,
	SubsurfaceSigmaAbsorption = 40,
	SubsurfaceSigmaScattering = 41,
	SubsurfacePhaseEvalHg = 42,
	SubsurfacePhaseSampleHgCos = 43,
	SubsurfacePhaseSampleWo = 44,
	DwivediDiffusionLength = 45,
	DwivediPhaseEval = 46,
	DwivediPhaseSampleCos = 47,
	SubsurfaceEntrySampleWo = 48,
	SubsurfaceEntryPdf = 49,
};

func spectrumRgb(rgb: f32v3) -> vkof::Spectrum {
	return vkof::Spectrum.fromRgb(rgb, vkof::wavelengthSampleUniform(0.5f));
}

func rgbOf(s: vkof::Spectrum) -> f32v3 {
	return s.asRgb(vkof::wavelengthSampleUniform(0.5f));
}

// transmission inputs: a = (ωi, isInsideMedium), b = (ωo, dispersedIor)
// c = (transmissionColor, transmissionDepth)
// d = (specularRoughness, anisotropy, anisotropyRotation, geometryThinWalled)
func transmissionParams(c: f32v4, d: f32v4) -> vkof::MaterialBackend_OpenPbr.Params {
	var params: vkof::MaterialBackend_OpenPbr.Params;
	params.transmissionColor = spectrumRgb(c.xyz);
	params.transmissionDepth = c.w;
	params.specularRoughness = d.x;
	params.specularRoughnessAnisotropy = d.y;
	params.specularRoughnessAnisotropyRotation = d.z;
	params.geometryThinWalled = d.w;
	return params;
}

func transmissionMaterial(c: f32v4, d: f32v4) -> OpenPbrMaterial {
	var mat: OpenPbrMaterial;
	mat.transmissionColor = c.xyz;
	mat.transmissionDepth = c.w;
	mat.specularRoughness = d.x;
	mat.specularRoughnessAnisotropy = d.y;
	mat.specularRoughnessAnisotropyRotation = d.z;
	mat.geometryThinWalled = d.w;
	return mat;
}

// test-only reference: zeltner 2022 fetchCoeffs (repo-references/ltc-sheen,
// pbrt-v3 sheenltc.cpp), bilinear as an explicit four-weight sum
func zeltnerReference(handle: u32, μ: f32, roughness: f32) -> f32v3 {
	let x: f32 = clamp(μ, 0.0f, 0.99999f) * 31.0f;
	let y: f32 = clamp(roughness, 0.0f, 0.99999f) * 31.0f;
	let x0: i32 = i32(x);
	let y0: i32 = i32(y);
	let fx: f32 = x - f32(x0);
	let fy: f32 = y - f32(y0);
	let table: Sampler2D = vkofTextures[handle];
	return (
		table.Load(i32v3(x0, y0, 0)).rgb * ((1.0f - fx) * (1.0f - fy))
		+ table.Load(i32v3(x0 + 1, y0, 0)).rgb * (fx * (1.0f - fy))
		+ table.Load(i32v3(x0, y0 + 1, 0)).rgb * ((1.0f - fx) * fy)
		+ table.Load(i32v3(x0 + 1, y0 + 1, 0)).rgb * (fx * fy)
	);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 threadId : SV_DispatchThreadID,
) {
	let index: u32 = threadId.x;
	if (index >= pc.count) { return; }
	let inputs: f32v4 * = ((f32v4 *)(pc.inVa)) + index * 9u;
	let a: f32v4 = inputs[0];
	let b: f32v4 = inputs[1];
	let c: f32v4 = inputs[2];
	let d: f32v4 = inputs[3];
	let e: f32v4 = inputs[4];
	let f: f32v4 = inputs[5];
	let g: f32v4 = inputs[6];
	let h: f32v4 = inputs[7];
	let k: f32v4 = inputs[8];
	var outNew: f32v4 = f32v4(0.0f);
	var outOld: f32v4 = f32v4(0.0f);
	var outMx: f32v4 = f32v4(0.0f);

	let tanX: f32v3 = f32v3(1.0f, 0.0f, 0.0f);
	let tanY: f32v3 = f32v3(0.0f, 1.0f, 0.0f);
	let nor: f32v3 = f32v3(0.0f, 0.0f, 1.0f);

	switch (ParityFunction(pc.function)) {
		case ParityFunction::FresnelDielectric: {
			// a = (cosθ, η)
			outNew.x = vkof::bsdfFresnelDielectric(a.x, a.y);
			outOld.x = ::utilFresnelDielectric(a.x, a.y);
			outMx.x = mxFresnelDielectric(a.x, a.y);
		} break;
		case ParityFunction::FresnelMetallic: {
			// a = (f0.rgb, specularWeight), b = (specularColor.rgb, cosθ)
			outNew.xyz = rgbOf(
				vkof::bsdfFresnelMetallic(
					spectrumRgb(a.xyz), a.w, spectrumRgb(b.xyz), b.w
				)
			);
			outOld.xyz = ::utilFresnelMetallic(a.xyz, a.w, b.xyz, b.w);
		} break;
		case ParityFunction::GgxDistributionAniso: {
			// a = h, b = α
			outNew.x = vkof::bsdfMicrofacetGgxDistributionAniso(a.xyz, b.xy);
			outOld.x = ::utilMicrofacetGgxDistributionAniso(a.xyz, b.xy);
			outMx.x = mx_ggx_NDF(a.xyz, b.xy);
		} break;
		case ParityFunction::SmithVisibilityAniso: {
			// a = ωi, b = ωo, c = α
			outNew.x = (
				vkof::bsdfMicrofacetSmithGgxVisibilityAniso(a.xyz, b.xyz, c.xy)
			);
			outOld.x = (
				::utilMicrofacetSmithGgxVisibilityAniso(a.xyz, b.xyz, c.xy)
			);
		} break;
		case ParityFunction::SmithG1Aniso: {
			// a = ω, b = α
			outNew.x = vkof::bsdfMicrofacetSmithG1Aniso(a.xyz, b.xy);
			outOld.x = ::utilMicrofacetSmithG1Aniso(a.xyz, b.xy);
			// materialx's G1 is isotropic; only meaningful when α.x == α.y
			outMx.x = mx_ggx_smith_G1(a.z, b.x);
		} break;
		case ParityFunction::SampleGgxBoundedWoAniso: {
			// a = ωi, b = α, c = ξ
			outNew.xyz = (
				vkof::bsdfMicrofacetSampleGgxBoundedWoAniso(
					tanX, tanY, nor, a.xyz, b.xy, c.xy
				)
			);
			outOld.xyz = (
				::utilMicrofacetSampleGgxBoundedWoAniso(
					tanX, tanY, nor, a.xyz, b.xy, c.xy
				)
			);
		} break;
		case ParityFunction::SampleGgxVndfAniso: {
			// a = ωi, b = α, c = ξ
			outNew.xyz = (
				vkof::bsdfMicrofacetSampleGgxVndfAniso(
					tanX, tanY, nor, a.xyz, b.xy, c.xy
				)
			);
			outOld.xyz = (
				::utilMicrofacetSampleGgxVndfAniso(
					tanX, tanY, nor, a.xyz, b.xy, c.xy
				)
			);
		} break;
		case ParityFunction::GgxBoundedReflectPdfAniso: {
			// a = ωi, b = ωo, c = α
			outNew.x = (
				vkof::bsdfMicrofacetGgxBoundedReflectPdfAniso(
					tanX, tanY, nor, a.xyz, b.xyz, c.xy
				)
			);
			outOld.x = (
				::utilMicrofacetGgxBoundedReflectPdfAniso(
					tanX, tanY, nor, a.xyz, b.xyz, c.xy
				)
			);
		} break;
		case ParityFunction::GgxDirectionalAlbedo: {
			// a = (μ, α, f0, f90)
			outNew.x = vkof::bsdfMicrofacetGgxDirectionalAlbedo(a.x, a.y, a.z, a.w);
			outOld.x = ::utilMicrofacetGgxDirectionalAlbedo(a.x, a.y, a.z, a.w);
			outMx.x = mx_ggx_dir_albedo_analytic(a.x, a.y, a.z, a.w);
		} break;
		case ParityFunction::DielectricEnergyCompensate: {
			// a = (μ, roughness, f0)
			outNew.x = vkof::bsdfMicrofacetDielectricEnergyCompensate(a.x, a.y, a.z);
			outOld.x = ::utilMicrofacetDielectricEnergyCompensate(a.x, a.y, a.z);
		} break;
		case ParityFunction::DielectricAlbedo: {
			// a = (μ, roughness, f0)
			outNew.x = vkof::bsdfMicrofacetDielectricAlbedo(a.x, a.y, a.z);
			outOld.x = ::utilMicrofacetDielectricAlbedo(a.x, a.y, a.z);
		} break;
		case ParityFunction::RoughnessAlpha: {
			// a = (roughness, anisotropy)
			outNew.xy = vkof::bsdfOpenPbrRoughnessAlpha(a.x, a.y);
			outOld.xy = ::openPbrRoughnessAlpha(a.x, a.y);
		} break;
		case ParityFunction::CoatRoughenedRoughness: {
			// a = (baseRoughness, coatRoughness, coatWeight)
			outNew.x = vkof::bsdfOpenPbrCoatRoughenedRoughness(a.x, a.y, a.z);
			outOld.x = ::openPbrCoatRoughenedRoughness(a.x, a.y, a.z);
		} break;
		case ParityFunction::CoatAffectedIor: {
			// a = (specularIor, coatIor, coatWeight)
			outNew.x = vkof::bsdfOpenPbrCoatAffectedIor(a.x, a.y, a.z);
			outOld.x = ::openPbrCoatAffectedIor(a.x, a.y, a.z);
		} break;
		case ParityFunction::EffectiveIor: {
			// a = (ior, weight)
			outNew.x = vkof::bsdfOpenPbrEffectiveIor(a.x, a.y);
			outOld.x = ::openPbrEffectiveIor(a.x, a.y);
		} break;
		case ParityFunction::SpecularEffectiveIor: {
			// a = (specularIor, coatIor, coatWeight, specularWeight)
			var params: vkof::MaterialBackend_OpenPbr.Params;
			params.specularIor = a.x;
			params.coatIor = a.y;
			params.coatWeight = a.z;
			params.specularWeight = a.w;
			var mat: OpenPbrMaterial;
			mat.specularIor = a.x;
			mat.coatIor = a.y;
			mat.coatWeight = a.z;
			mat.specularWeight = a.w;
			outNew.x = vkof::bsdfOpenPbrSpecularEffectiveIor(params);
			outOld.x = ::openPbrSpecularEffectiveIor(mat);
		} break;
		case ParityFunction::IorFromF0: {
			// a = f0.rgb
			outNew.xyz = rgbOf(vkof::bsdfOpenPbrIorFromF0(spectrumRgb(a.xyz)));
			outOld.xyz = ::openPbrIorFromF0(a.xyz);
		} break;
		case ParityFunction::HalfVectorLocal: {
			// a = ωi, b = ωo; w = valid
			var hNew: f32v3;
			var hOld: f32v3;
			let validNew: bool = vkof::bsdfHalfVectorLocal(a.xyz, b.xyz, hNew);
			let validOld: bool = ::utilHalfVectorLocal(a.xyz, b.xyz, hOld);
			outNew = f32v4(hNew, validNew ? 1.0f : 0.0f);
			outOld = f32v4(hOld, validOld ? 1.0f : 0.0f);
		} break;
		case ParityFunction::FresnelDielectricPhase: {
			// a = (cosθi, cosθt, ηi, ηt); out = (Rp, Rs, φp, φs)
			var Rp: f32; var Rs: f32; var φp: f32; var φs: f32;
			vkof::bsdfFresnelDielectricPhase(a.x, a.y, a.z, a.w, Rp, Rs, φp, φs);
			outNew = f32v4(Rp, Rs, φp, φs);
			::utilFresnelDielectricPhase(a.x, a.y, a.z, a.w, Rp, Rs, φp, φs);
			outOld = f32v4(Rp, Rs, φp, φs);
		} break;
		case ParityFunction::FresnelConductorPhase: {
			// a = (cosθi, cosθt, ηi, ηt), b.x = κt; out = (Rp, Rs, φp, φs)
			var Rp: f32; var Rs: f32; var φp: f32; var φs: f32;
			vkof::bsdfFresnelConductorPhase(a.x, a.y, a.z, a.w, b.x, Rp, Rs, φp, φs);
			outNew = f32v4(Rp, Rs, φp, φs);
			::utilFresnelConductorPhase(a.x, a.y, a.z, a.w, b.x, Rp, Rs, φp, φs);
			outOld = f32v4(Rp, Rs, φp, φs);
		} break;
		case ParityFunction::EvaluateXyzSensitivity: {
			// a = (opd, shift)
			outNew.xyz = vkof::bsdfEvaluateXyzSensitivity(a.x, a.y);
			outOld.xyz = ::utilEvaluateXyzSensitivity(a.x, a.y);
		} break;
		case ParityFunction::ThinfilmFresnel: {
			// a = (thinFilmThickness, thinFilmIor, cosθi), b = ηBase, c = κBase
			var params: vkof::MaterialBackend_OpenPbr.Params;
			params.thinFilmThickness = a.x;
			params.thinFilmIor = a.y;
			var mat: OpenPbrMaterial;
			mat.thinFilmThickness = a.x;
			mat.thinFilmIor = a.y;
			outNew.xyz = vkof::bsdfOpenPbrThinfilmFresnel(params, a.z, b.xyz, c.xyz);
			outOld.xyz = ::openPbrThinfilmFresnel(mat, a.z, b.xyz, c.xyz);
		} break;
		case ParityFunction::DielectricSpecularEvaluateF: {
			// a = ωi, b = ωo, a.w = ηRel
			// c = (specularRoughness, coatRoughness, coatWeight, anisotropy)
			// d = (anisotropyRotation, thinFilmWeight, thinFilmIor, thinFilmThickness)
			var params: vkof::MaterialBackend_OpenPbr.Params;
			params.specularRoughness = c.x;
			params.coatRoughness = c.y;
			params.coatWeight = c.z;
			params.specularRoughnessAnisotropy = c.w;
			params.specularRoughnessAnisotropyRotation = d.x;
			params.thinFilmWeight = d.y;
			params.thinFilmIor = d.z;
			params.thinFilmThickness = d.w;
			var mat: OpenPbrMaterial;
			mat.specularRoughness = c.x;
			mat.coatRoughness = c.y;
			mat.coatWeight = c.z;
			mat.specularRoughnessAnisotropy = c.w;
			mat.specularRoughnessAnisotropyRotation = d.x;
			mat.thinFilmWeight = d.y;
			mat.thinFilmIor = d.z;
			mat.thinFilmThickness = d.w;
			var frameNew: vkof::ShadingFrame;
			frameNew.tanX = tanX;
			frameNew.tanY = tanY;
			frameNew.nor = nor;
			var frameOld: ::ShadingFrame;
			frameOld.tanX = tanX;
			frameOld.tanY = tanY;
			frameOld.nor = nor;
			outNew.xyz = rgbOf(
				vkof::bsdfOpenPbrDielectricSpecularEvaluateF(
					params, frameNew, a.xyz, b.xyz, a.w,
					vkof::wavelengthSampleUniform(0.5f)
				)
			);
			outOld.xyz = (
				::openPbrDielectricSpecularEvaluateF(
					mat, frameOld, a.xyz, b.xyz, a.w
				)
			);
		} break;
		case ParityFunction::FresnelMetallicEvaluateF: {
			// a = (ωi, baseWeight), b = (ωo, baseMetalness)
			// c = (baseColor, specularWeight), d = (specularColor, specularRoughness)
			// e = (coatRoughness, coatWeight, anisotropy, anisotropyRotation)
			// f = (thinFilmWeight, thinFilmIor, thinFilmThickness)
			var params: vkof::MaterialBackend_OpenPbr.Params;
			params.baseWeight = a.w;
			params.baseMetalness = b.w;
			params.baseColor = spectrumRgb(c.xyz);
			params.specularWeight = c.w;
			params.specularColor = spectrumRgb(d.xyz);
			params.specularRoughness = d.w;
			params.coatRoughness = e.x;
			params.coatWeight = e.y;
			params.specularRoughnessAnisotropy = e.z;
			params.specularRoughnessAnisotropyRotation = e.w;
			params.thinFilmWeight = f.x;
			params.thinFilmIor = f.y;
			params.thinFilmThickness = f.z;
			var mat: OpenPbrMaterial;
			mat.baseWeight = a.w;
			mat.baseMetalness = b.w;
			mat.baseColor = c.xyz;
			mat.specularWeight = c.w;
			mat.specularColor = d.xyz;
			mat.specularRoughness = d.w;
			mat.coatRoughness = e.x;
			mat.coatWeight = e.y;
			mat.specularRoughnessAnisotropy = e.z;
			mat.specularRoughnessAnisotropyRotation = e.w;
			mat.thinFilmWeight = f.x;
			mat.thinFilmIor = f.y;
			mat.thinFilmThickness = f.z;
			var frameNew: vkof::ShadingFrame;
			frameNew.tanX = tanX;
			frameNew.tanY = tanY;
			frameNew.nor = nor;
			var frameOld: ::ShadingFrame;
			frameOld.tanX = tanX;
			frameOld.tanY = tanY;
			frameOld.nor = nor;
			outNew.xyz = rgbOf(
				vkof::bsdfOpenPbrFresnelMetallicEvaluateF(
					params, frameNew, a.xyz, b.xyz,
					vkof::wavelengthSampleUniform(0.5f)
				)
			);
			outOld.xyz = (
				::openPbrFresnelMetallicEvaluateF(mat, frameOld, a.xyz, b.xyz)
			);
		} break;
		case ParityFunction::OrenNayerEnergyCompensate: {
			// a = (μ, σ)
			outNew.x = vkof::bsdfOrenNayerEnergyCompensate(a.x, a.y);
			outOld.x = ::utilOrenNayerEnergyCompensate(a.x, a.y);
			// materialx: exact E_FON, ponder: portsmouth polynomial fit
			outMx.x = mxOrenNayarFujiiDiffuseDirAlbedo(a.x, a.y);
		} break;
		case ParityFunction::GlossyDiffuseEvaluateF: {
			// a = (ωi, baseWeight), b = (ωo, baseDiffuseRoughness), c = baseColor
			var params: vkof::MaterialBackend_OpenPbr.Params;
			params.baseWeight = a.w;
			params.baseDiffuseRoughness = b.w;
			params.baseColor = spectrumRgb(c.xyz);
			var mat: OpenPbrMaterial;
			mat.baseWeight = a.w;
			mat.baseDiffuseRoughness = b.w;
			mat.baseColor = c.xyz;
			outNew.xyz = rgbOf(
				vkof::bsdfOpenPbrGlossyDiffuseOrenNayerEvaluateF(
					params, nor, a.xyz, b.xyz, vkof::wavelengthSampleUniform(0.5f)
				)
			);
			outOld.xyz = (
				::openPbrGlossyDiffuseOrenNayerEvaluateF(mat, nor, a.xyz, b.xyz)
			);
			// materialx's brdf carries no 1/π and no weight
			outMx.xyz = (
				mxOrenNayarCompensatedDiffuse(
					dot(nor, a.xyz), dot(nor, b.xyz), dot(a.xyz, b.xyz), b.w, c.xyz
				)
				* (a.w / 3.14159265358979f)
			);
		} break;
		case ParityFunction::ThinWalledSubsurfaceEvaluateF: {
			// a = (ωi, baseDiffuseRoughness), b = (ωo, scatterAnisotropy)
			// c = subsurfaceColor
			var params: vkof::MaterialBackend_OpenPbr.Params;
			params.baseDiffuseRoughness = a.w;
			params.subsurfaceScatterAnisotropy = b.w;
			params.subsurfaceColor = spectrumRgb(c.xyz);
			var mat: OpenPbrMaterial;
			mat.baseDiffuseRoughness = a.w;
			mat.subsurfaceScatterAnisotropy = b.w;
			mat.subsurfaceColor = c.xyz;
			outNew.xyz = rgbOf(
				vkof::bsdfOpenPbrThinWalledSubsurfaceEvaluateF(
					params, nor, a.xyz, b.xyz, vkof::wavelengthSampleUniform(0.5f)
				)
			);
			outOld.xyz = (
				::openPbrThinWalledSubsurfaceEvaluateF(mat, nor, a.xyz, b.xyz)
			);
		} break;
		case ParityFunction::ThinWalledSubsurfacePdf: {
			// a = (dotNorWo, anisotropy)
			outNew.x = vkof::bsdfOpenPbrThinWalledSubsurfacePdf(a.x, a.y);
			outOld.x = ::openPbrThinWalledSubsurfacePdf(a.x, a.y);
		} break;
		case ParityFunction::GlossyDiffusePdf: {
			// a = (dotNorWo)
			outNew.x = vkof::bsdfOpenPbrGlossyDiffusePdf(a.x);
			outOld.x = ::openPbrGlossyDiffusePdf(a.x);
		} break;
		case ParityFunction::ZeltnerFuzzLookup: {
			// a = (μ, roughness)
			outNew.xyz = (
				vkof::bsdfZeltnerFuzzLookup(pc.zeltnerLtcParamHandle, a.x, a.y)
			);
			var tables: MaterialTableHandles;
			tables.kullaContyEnergy = 0u;
			tables.zeltnerLtcParam = pc.zeltnerLtcParamHandle;
			outOld.xyz = ::utilZeltnerFuzzLookup(tables, a.x, a.y);
			outMx.xyz = zeltnerReference(pc.zeltnerLtcParamHandle, a.x, a.y);
		} break;
		case ParityFunction::FuzzEvaluateF: {
			// a = (ωi, fuzzWeight), b = (ωo, fuzzRoughness)
			// c = fuzzColor, d = coatedBase
			var params: vkof::MaterialBackend_OpenPbr.Params;
			params.fuzzWeight = a.w;
			params.fuzzRoughness = b.w;
			params.fuzzColor = spectrumRgb(c.xyz);
			var mat: OpenPbrMaterial;
			mat.fuzzWeight = a.w;
			mat.fuzzRoughness = b.w;
			mat.fuzzColor = c.xyz;
			var tables: MaterialTableHandles;
			tables.kullaContyEnergy = 0u;
			tables.zeltnerLtcParam = pc.zeltnerLtcParamHandle;
			outNew.xyz = rgbOf(
				vkof::bsdfOpenPbrFuzzEvaluateF(
					pc.zeltnerLtcParamHandle, params, spectrumRgb(d.xyz),
					nor, a.xyz, b.xyz, vkof::wavelengthSampleUniform(0.5f)
				)
			);
			outOld.xyz = (
				::openPbrFuzzEvaluateF(tables, mat, d.xyz, nor, a.xyz, b.xyz)
			);
			// test-only reference lobe, zeltner 2022 eq 1 with the reference lookup
			if (a.w > 0.0f) {
				let coeffs: f32v3 = (
					zeltnerReference(pc.zeltnerLtcParamHandle, dot(nor, a.xyz), b.w)
				);
				var refTanX: f32v3;
				var refTanY: f32v3;
				vkof::bsdfFuzzFrame(nor, a.xyz, refTanX, refTanY);
				let t: f32v3 = f32v3(
					coeffs.x * dot(b.xyz, refTanX) + coeffs.y * dot(nor, b.xyz),
					coeffs.x * dot(b.xyz, refTanY),
					dot(nor, b.xyz)
				);
				let lenT2: f32 = max(dot(t, t), 1e-8f);
				let lobe: f32 = coeffs.x * coeffs.x / (3.14159265358979f * lenT2 * lenT2);
				outMx.xyz = (
					a.w * c.xyz * coeffs.z * lobe + (1.0f - a.w * coeffs.z) * d.xyz
				);
			} else {
				outMx.xyz = d.xyz;
			}
		} break;
		case ParityFunction::FuzzPdf: {
			// a = (ωi, fuzzA), b = (ωo, fuzzB)
			outNew.x = vkof::bsdfOpenPbrFuzzPdf(nor, a.xyz, b.xyz, a.w, b.w);
			outOld.x = ::openPbrFuzzPdf(nor, a.xyz, b.xyz, a.w, b.w);
		} break;
		case ParityFunction::FuzzSampleWo: {
			// a = (ωi, fuzzA), b = (ξ, fuzzB)
			outNew.xyz = vkof::bsdfOpenPbrFuzzSampleWo(nor, a.xyz, a.w, b.w, b.xy);
			outOld.xyz = ::openPbrFuzzSampleWo(nor, a.xyz, a.w, b.w, b.xy);
		} break;
		case ParityFunction::CoatEvaluateF: {
			// a = (ωi, isInsideMedium), b = (ωo, coatIor)
			// c = (coatColor, coatWeight), d = (baseColor, coatRoughness)
			// e = (subsurfaceColor, coatRoughnessAnisotropy)
			// f = (transmissionColor, coatDarkening)
			// g = (baseSubstrate, specularIor)
			// h = (specularWeight, specularRoughness, baseMetalness, baseWeight)
			// k = (subsurfaceWeight, transmissionWeight)
			var params: vkof::MaterialBackend_OpenPbr.Params;
			params.coatIor = b.w;
			params.coatColor = spectrumRgb(c.xyz);
			params.coatWeight = c.w;
			params.baseColor = spectrumRgb(d.xyz);
			params.coatRoughness = d.w;
			params.subsurfaceColor = spectrumRgb(e.xyz);
			params.coatRoughnessAnisotropy = e.w;
			params.transmissionColor = spectrumRgb(f.xyz);
			params.coatDarkening = f.w;
			params.specularIor = g.w;
			params.specularWeight = h.x;
			params.specularRoughness = h.y;
			params.baseMetalness = h.z;
			params.baseWeight = h.w;
			params.subsurfaceWeight = k.x;
			params.transmissionWeight = k.y;
			var mat: OpenPbrMaterial;
			mat.coatIor = b.w;
			mat.coatColor = c.xyz;
			mat.coatWeight = c.w;
			mat.baseColor = d.xyz;
			mat.coatRoughness = d.w;
			mat.subsurfaceColor = e.xyz;
			mat.coatRoughnessAnisotropy = e.w;
			mat.transmissionColor = f.xyz;
			mat.coatDarkening = f.w;
			mat.specularIor = g.w;
			mat.specularWeight = h.x;
			mat.specularRoughness = h.y;
			mat.baseMetalness = h.z;
			mat.baseWeight = h.w;
			mat.subsurfaceWeight = k.x;
			mat.transmissionWeight = k.y;
			var frameNew: vkof::ShadingFrame;
			frameNew.tanX = tanX;
			frameNew.tanY = tanY;
			frameNew.nor = nor;
			var frameOld: ::ShadingFrame;
			frameOld.tanX = tanX;
			frameOld.tanY = tanY;
			frameOld.nor = nor;
			var tables: MaterialTableHandles;
			tables.kullaContyEnergy = 0u;
			tables.zeltnerLtcParam = 0u;
			let isInside: bool = a.w > 0.5f;
			outNew.xyz = rgbOf(
				vkof::bsdfOpenPbrCoatEvaluateF(
					params, frameNew, a.xyz, b.xyz, spectrumRgb(g.xyz), isInside,
					vkof::wavelengthSampleUniform(0.5f)
				)
			);
			outOld.xyz = (
				::openPbrCoatEvaluateF(
					tables, mat, frameOld, a.xyz, b.xyz, g.xyz, isInside
				)
			);
		} break;
		case ParityFunction::TransmissionEvaluateF: {
			var frameNew: vkof::ShadingFrame;
			frameNew.tanX = tanX;
			frameNew.tanY = tanY;
			frameNew.nor = nor;
			var frameOld: ::ShadingFrame;
			frameOld.tanX = tanX;
			frameOld.tanY = tanY;
			frameOld.nor = nor;
			outNew.xyz = rgbOf(
				vkof::bsdfOpenPbrTransmissionEvaluateF(
					transmissionParams(c, d), frameNew, a.xyz, b.xyz, a.w > 0.5f,
					b.w, vkof::wavelengthSampleUniform(0.5f)
				)
			);
			outOld.xyz = (
				::openPbrTransmissionEvaluateF(
					transmissionMaterial(c, d), frameOld, a.xyz, b.xyz, a.w > 0.5f,
					b.w
				)
			);
		} break;
		case ParityFunction::TransmissionPdf: {
			var frameNew: vkof::ShadingFrame;
			frameNew.tanX = tanX;
			frameNew.tanY = tanY;
			frameNew.nor = nor;
			var frameOld: ::ShadingFrame;
			frameOld.tanX = tanX;
			frameOld.tanY = tanY;
			frameOld.nor = nor;
			outNew.x = (
				vkof::bsdfOpenPbrTransmissionPdf(
					transmissionParams(c, d), frameNew, a.xyz, b.xyz, a.w > 0.5f, b.w
				)
			);
			outOld.x = (
				::openPbrTransmissionPdf(
					transmissionMaterial(c, d), frameOld, a.xyz, b.xyz, a.w > 0.5f,
					b.w
				)
			);
		} break;
		case ParityFunction::TransmissionTirReflectPdf: {
			var frameNew: vkof::ShadingFrame;
			frameNew.tanX = tanX;
			frameNew.tanY = tanY;
			frameNew.nor = nor;
			var frameOld: ::ShadingFrame;
			frameOld.tanX = tanX;
			frameOld.tanY = tanY;
			frameOld.nor = nor;
			outNew.x = (
				vkof::bsdfOpenPbrTransmissionTirReflectPdf(
					transmissionParams(c, d), frameNew, a.xyz, b.xyz, a.w > 0.5f, b.w
				)
			);
			outOld.x = (
				::openPbrTransmissionTirReflectPdf(
					transmissionMaterial(c, d), frameOld, a.xyz, b.xyz, a.w > 0.5f,
					b.w
				)
			);
		} break;
		case ParityFunction::DispersionIorRgb: {
			// a = (specularIor, coatIor, coatWeight, specularWeight)
			// b = (dispersionScale, abbeNumber)
			var params: vkof::MaterialBackend_OpenPbr.Params;
			params.specularIor = a.x;
			params.coatIor = a.y;
			params.coatWeight = a.z;
			params.specularWeight = a.w;
			params.transmissionDispersionScale = b.x;
			params.transmissionDispersionAbbeNumber = b.y;
			var mat: OpenPbrMaterial;
			mat.specularIor = a.x;
			mat.coatIor = a.y;
			mat.coatWeight = a.z;
			mat.specularWeight = a.w;
			mat.transmissionDispersionScale = b.x;
			mat.transmissionDispersionAbbeNumber = b.y;
			outNew.xyz = vkof::bsdfOpenPbrDispersionIorRgb(params);
			outOld.xyz = ::openPbrDispersionIorRgb(mat);
		} break;
		case ParityFunction::SubsurfaceDiffusionReflectance: {
			outNew.xyz = f32v3(
				vkof::bsdfOpenPbrSubsurfaceDiffusionReflectance(a.x),
				vkof::bsdfOpenPbrSubsurfaceDiffusionReflectance(a.y),
				vkof::bsdfOpenPbrSubsurfaceDiffusionReflectance(a.z)
			);
			outOld.xyz = ::openPbrSubsurfaceDiffusionReflectance(a.xyz);
		} break;
		case ParityFunction::SubsurfaceInvertDiffusionReflectance: {
			outNew.xyz = f32v3(
				vkof::bsdfOpenPbrSubsurfaceInvertDiffusionReflectance(a.x),
				vkof::bsdfOpenPbrSubsurfaceInvertDiffusionReflectance(a.y),
				vkof::bsdfOpenPbrSubsurfaceInvertDiffusionReflectance(a.z)
			);
			outOld.xyz = ::openPbrSubsurfaceInvertDiffusionReflectance(a.xyz);
		} break;
		case ParityFunction::SubsurfaceSigmaAbsorption:
		case ParityFunction::SubsurfaceSigmaScattering: {
			// a = (albedo, anisotropy), b = radius
			var σaSpectrum: vkof::Spectrum;
			var σsSpectrum: vkof::Spectrum;
			vkof::bsdfOpenPbrSubsurfaceAlbedoToSigma(
				spectrumRgb(a.xyz),
				spectrumRgb(b.xyz),
				a.w,
				vkof::wavelengthSampleUniform(0.5f),
				σaSpectrum,
				σsSpectrum
			);
			let σaNew: f32v3 = rgbOf(σaSpectrum);
			let σsNew: f32v3 = rgbOf(σsSpectrum);
			var σaOld: f32v3;
			var σsOld: f32v3;
			::openPbrSubsurfaceAlbedoToSigma(a.xyz, b.xyz, a.w, σaOld, σsOld);
			let absorption: bool = (
				ParityFunction(pc.function) == ParityFunction::SubsurfaceSigmaAbsorption
			);
			outNew.xyz = absorption ? σaNew : σsNew;
			outOld.xyz = absorption ? σaOld : σsOld;
		} break;
		case ParityFunction::SubsurfacePhaseEvalHg: {
			// a = (cosθ, g)
			outNew.x = vkof::bsdfOpenPbrSubsurfacePhaseEvalHg(a.x, a.y);
			outOld.x = ::openPbrSubsurfacePhaseEvalHg(a.x, a.y);
		} break;
		case ParityFunction::SubsurfacePhaseSampleHgCos: {
			// a = (u, g)
			outNew.x = vkof::bsdfOpenPbrSubsurfacePhaseSampleHgCos(a.x, a.y);
			outOld.x = ::openPbrSubsurfacePhaseSampleHgCos(a.x, a.y);
		} break;
		case ParityFunction::SubsurfacePhaseSampleWo: {
			// a = (ωi, g), b = ξ
			outNew.xyz = vkof::bsdfOpenPbrSubsurfacePhaseSampleWo(a.xyz, a.w, b.xy);
			outOld.xyz = ::openPbrSubsurfacePhaseSampleWo(a.xyz, a.w, b.xy);
		} break;
		case ParityFunction::DwivediDiffusionLength: {
			outNew.x = vkof::bsdfOpenPbrDwivediDiffusionLength(a.x);
			outOld.x = ::openPbrDwivediDiffusionLength(a.x);
		} break;
		case ParityFunction::DwivediPhaseEval: {
			// a = (L, phaseLog, cosθ)
			outNew.x = vkof::bsdfOpenPbrDwivediPhaseEval(a.x, a.y, a.z);
			outOld.x = ::openPbrDwivediPhaseEval(a.x, a.y, a.z);
		} break;
		case ParityFunction::DwivediPhaseSampleCos: {
			// a = (L, phaseLog, ξ)
			outNew.x = vkof::bsdfOpenPbrDwivediPhaseSampleCos(a.x, a.y, a.z);
			outOld.x = ::openPbrDwivediPhaseSampleCos(a.x, a.y, a.z);
		} break;
		case ParityFunction::SubsurfaceEntrySampleWo: {
			// a = ξ
			outNew.xyz = vkof::bsdfOpenPbrSubsurfaceEntrySampleWo(nor, a.xy);
			outOld.xyz = ::openPbrSubsurfaceEntrySampleWo(nor, a.xy);
		} break;
		case ParityFunction::SubsurfaceEntryPdf: {
			// a = (dotNorWo)
			outNew.x = vkof::bsdfOpenPbrSubsurfaceEntryPdf(a.x);
			outOld.x = ::openPbrSubsurfaceEntryPdf(a.x);
		} break;
		default: break;
	}

	let outputs: f32v4 * = ((f32v4 *)(pc.outVa)) + index * 3u;
	outputs[0] = outNew;
	outputs[1] = outOld;
	outputs[2] = outMx;
}
