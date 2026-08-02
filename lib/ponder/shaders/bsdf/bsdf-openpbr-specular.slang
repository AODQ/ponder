implementing bsdf;

import types;
import util;
import material;

// directions in this file point away from the surface: ωi here is the
// path's -ωi (see units.slang); Bsdf_OpenPbr negates it once at entry

namespace vkof {

// microfacet dielectric interface reflection, shared by the opaque base and
// the translucent base
public func bsdfOpenPbrDielectricSpecularEvaluateF(
	mat: MaterialBackend_OpenPbr.Params,
	frame: ShadingFrame,
	ωi: f32v3,
	ωo: f32v3,
	ηRel: f32,
	λ: Wavelengths,
) -> vkof::Spectrum {
	/*
	   typical microfacet;
	   f_{dielectric} =
	     mf_f * mf_g * mf_v
	     / (4 * (\omega_wi \cdot n) * (\omega_wo \cdot n))
	   \\
	   where,
	     \eta = relative ior with the specular weight folded in
	*/
	let roughenedRoughness: f32 = (
		bsdfOpenPbrCoatRoughenedRoughness(
			mat.specularRoughness, mat.coatRoughness, mat.coatWeight
		)
	);
	let α: f32v2 = (
		bsdfOpenPbrRoughnessAlpha(
			roughenedRoughness, mat.specularRoughnessAnisotropy
		)
	);
	// gltf interop anisotropy rotation (not part of openpbr)
	let specFrame: ShadingFrame = (
		shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
	);
	// local-frame directions for the anisotropic D and V
	let ωiLocal: f32v3 = (
		f32v3(
			dot(ωi, specFrame.tanX),
			dot(ωi, specFrame.tanY),
			dot(ωi, specFrame.nor)
		)
	);
	let ωoLocal: f32v3 = (
		f32v3(
			dot(ωo, specFrame.tanX),
			dot(ωo, specFrame.tanY),
			dot(ωo, specFrame.nor)
		)
	);
	var hLocal: f32v3;
	if (!bsdfHalfVectorLocal(ωiLocal, ωoLocal, hLocal)) {
		return vkof::Spectrum.zero();
	}
	let dotHWi: f32 = dot(hLocal, ωiLocal);

	// negative dotHWi is a back-facing microfacet
	if (dotHWi <= 0.0f) {
		return vkof::Spectrum.zero();
	}

	// thin-film interference interface substitutes for dielectric fresnel
	// interface. metallic lobe gets its own tinted thin-film fresnel
	// elsewhere
	let dielectricFresnel: vkof::Spectrum = (
		vkof::Spectrum.one() * bsdfFresnelDielectric(dotHWi, ηRel)
	);
	// gltf KHR_materials_iridescence: iridescenceFactor is a lerp sadly
	let mfFresnel: vkof::Spectrum = (
		mat.thinFilmWeight > 0.0f
		? lerp(
			dielectricFresnel,
			bsdfOpenPbrThinfilmFresnel(
				mat,
				dotHWi,
				vkof::Spectrum.one() * ηRel,
				vkof::Spectrum.zero(),
				λ
			),
			mat.thinFilmWeight
		)
		: dielectricFresnel
	);
	let mfDistribution: f32 = (
		bsdfMicrofacetGgxDistributionAniso(hLocal, α)
	);
	let mfVisibility: f32 = (
		bsdfMicrofacetSmithGgxVisibilityAniso(ωiLocal, ωoLocal, α)
	);
	// turquin 2019 multiplicative energy compensation; f0 from the relative
	// ior, which is symmetric in \eta vs 1/\eta. the factor uses the view
	// cosine only. Isotropic so uses effective roughenss
	// \sqrt{\alpha_x \alpha_y}
	let iorRatio: f32 = (ηRel - 1.0f) / (ηRel + 1.0f);
	let f0: f32 = iorRatio * iorRatio;
	let energyCompensation: f32 = (
		bsdfMicrofacetDielectricEnergyCompensate(
			ωiLocal.z, sqrt(α.x * α.y), f0
		)
	);
	// the 1/(4 (\omega_wi \cdot n) (\omega_wo \cdot n)) term is already
	// folded into the smith visibility term
	// TODO multiply by specularweight right?
	return mfFresnel * (mfDistribution * mfVisibility * energyCompensation);
}

} // namespace vkof
