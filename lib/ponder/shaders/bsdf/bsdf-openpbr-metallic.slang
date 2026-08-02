implementing bsdf;

import types;
import util;
import material;

// directions in this file point away from the surface: ωi here is the
// path's -ωi (see units.slang); Bsdf_OpenPbr negates it once at entry

namespace vkof {

// openpbr conductor: f82-tint fresnel over the shared specular ggx lobe;
// f82-tint model from kutz, hasan & edmondson 2021
/*
	\begin{align*}
	&\textbf{(1) f82-tint conductor fresnel} \\
	&f_0 = base\_weight \cdot base\_color
		\tag{openpbr spec, metal}
	\\
	% schlick fresnel toward white at grazing
	&F_s(\mu) = f_0 + (1 - f_0)(1 - \mu)^5
		\tag{schlick 1994}
	\\
	% the edge tint is authored at \bar{\mu} = 1/7 (\theta \approx 82 deg)
	&\bar{\mu} = \tfrac{1}{7}\\
	&F_c = F_s(\bar{\mu}) (1 - S_w S_c)\\
	&F_d = \bar{\mu} (1 - \bar{\mu})^6\\
	% the correction term is zero at \mu = 0 and \mu = 1 and peaks at
	% \bar{\mu}, pulling the fresnel toward the authored target
	&F_{82}(\mu) = F_s(\mu) - \frac{\mu (1 - \mu)^6}{F_d} F_c\\
	\end{align*}
*/
public func bsdfOpenPbrFresnelMetallicEvaluateF(
	mat: MaterialBackend_OpenPbr.Params,
	frame: ShadingFrame,
	ωi: f32v3,
	ωo: f32v3,
	λ: Wavelengths,
) -> vkof::Spectrum {
	if (mat.baseMetalness <= 0.0f) {
		return vkof::Spectrum.zero();
	}

	let nor: f32v3 = frame.nor;
	let dotNorWi: f32 = max(dot(nor, ωi), 1e-5f);

	// f_0 = base\_weight \cdot base\_color
	let f0: vkof::Spectrum = (
		mat.baseWeight * mat.baseColor
	);

	// F_{82}(\mu)
	let f82: vkof::Spectrum = (
		bsdfFresnelMetallic(
			/*f0=*/ f0,
			/*specularWeight=*/ mat.specularWeight,
			/*specularColor=*/ mat.specularColor,
			/*cosθ=*/ dotNorWi
		)
	);

	// conductor shares the base specular roughness + anisotropy, widened
	// by a rough coat (openpbr spec, coat roughening)
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
	// gltf interop anisotropy rotation; openpbr does not have this I believe
	// but hacking it in for gltf interop :/
	let specFrame: ShadingFrame = (
		shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
	);
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
	let hValid: bool = bsdfHalfVectorLocal(ωiLocal, ωoLocal, hLocal);
	let D: f32 = (
		hValid ? bsdfMicrofacetGgxDistributionAniso(hLocal, α) : 0.0f
	);
	let V: f32 = (
		bsdfMicrofacetSmithGgxVisibilityAniso(ωiLocal, ωoLocal, α)
	);

	// thin-film interference over the metal f0
	// κ=0 as the f82-tint model has no real n/k data
	let dotHWi: f32 = dot(hLocal, ωiLocal);
	let ηBase: vkof::Spectrum = bsdfOpenPbrIorFromF0(f0);
	let mfFresnel: vkof::Spectrum = (
		mat.thinFilmWeight > 0.0f
		? lerp(
			f82,
			bsdfOpenPbrThinfilmFresnel(
				mat, dotHWi, ηBase, vkof::Spectrum.zero(), λ
			),
			mat.thinFilmWeight
		)
		: f82
	);

	// turquin 2019 conductor energy compensation, per channel with the
	// colored f_0; isotropic using effective roughness \sqrt{\alpha_x \alpha_y}
	let fAvg: vkof::Spectrum = f0 + (1.0f - f0) * (1.0f / 21.0f);
	let ess: f32 = (
		max(
			bsdfMicrofacetGgxDirectionalAlbedo(
				dotNorWi, sqrt(α.x * α.y), 1.0f, 1.0f
			),
			1e-4f
		)
	);
	let energyCompensation: vkof::Spectrum = (
		1.0f + fAvg * ((1.0f - ess) / ess)
	);

	return mfFresnel * (D * V) * energyCompensation;
}

} // namespace vkof
