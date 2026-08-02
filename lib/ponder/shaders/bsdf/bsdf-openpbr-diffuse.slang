implementing bsdf;

import types;
import util;
import material;

// directions in this file point away from the surface: ωi here is the
// path's -ωi (see units.slang); Bsdf_OpenPbr negates it once at entry

namespace vkof {

public func bsdfOrenNayerEnergyCompensate(
	μ: f32,
	r: f32,
) -> f32 {
	// portsmouth 2024 approximate form;
	// in future can either use a lookup table or a more accurate approximation
	let μcomp: f32 = 1.0f - μ;
	let g1: f32 = 0.0571085289f;
	let g2: f32 = 0.491881867f;
	let g3: f32 = -0.332181442f;
	let g4: f32 = 0.0714429953f;
	let gOverPi: f32 = μcomp * (g1 + μcomp * (g2 + μcomp * (g3 + μcomp * g4)));
	return (1.0f + r * gOverPi) / (1.0f + r * skFon1);
}

/*
	layer of dielectric gloss on top of an opaque diffuse slab.
	OpenPBR implements f_{diffuse} as energy-preserving Oren-Nayer-Fujii.

	M_{glossy-diffuse} = layer(S_{diffuse}, S_{gloss})\\
	S_{gloss} = Slab(f_{dielectric}, V_{dielectric})\\

	S_{diffuse} = Slab(f_{diffuse})\\

	f_{diffuse}(\omega_i, \omega_o) =
		f_{EON}(\omega_i, \omega_o) + f^{Fujii}_{EON}(\omega_i, \omega_o)
	\\
	F_{EON}(\omega_i, \omega_o) =
		\frac{w_d C}{\pi}
		(A(\sigma) + B(\sigma) \frac{s}{t})
	\\
	A(\sigma) = 1 / (1 + FON1 \sigma)\\
	B(\sigma) = \sigma A(\sigma)\\
	s = \omega_i \cdot \omega_o - (N \cdot \omega_i)(N \cdot \omega_o)\\
	t = s > 0 ? max(N \cdot \omega_i, N \cdot \omega_o) : 1\\
	f^{Fujii}_{EON}(\omega_i, \omega_o) =
		\frac{w_d \rho_{ms}}{\pi}
		(1 - \hat{E}_{ON}(\omega_i))
		(1 - \hat{E}_{ON}(\omega_o))
	\\
	\hat{E}_{ON}(\omega_i) = AF * (1.0 + FON2 * r)\\
	\rho_{ms} =
		\frac{C^2}{\pi}
		\frac
			{\langle \hat{E}_{ON} \rangle / (1 - \langle \hat{E}_{ON} \rangle)}
			{1 - C(1 - \langle \hat{E}_{ON} \rangle)}
	\\

	where,
		\rho_{ms} = multiple scattering on microfacet surface
		\sigma = base diffuse roughness
		\hat{E}_{ON} = energy compensation avg
		w_d = diffuse weight
		C = base color
		t is the maximum of the cosine of the incident and outgoing angles
		FON2 = energy conservation constant
		s is the azimuthal difference between \omega_i and \omega_o ;
			oren-nayer scattering surfaces retroreflects
*/
public func bsdfOpenPbrOrenNayerEvaluateF(
	color: vkof::Spectrum,
	weight: f32,
	σ: f32,
	nor: f32v3,
	ωi: f32v3,
	ωo: f32v3,
) -> vkof::Spectrum {

	// -- step 1: fujii oren-nayer scattering (EON paper, portsmouth 2024);

	let μi: f32 = dot(nor, ωi);
	let μo: f32 = dot(nor, ωo);

	// s = cos\phi sin\theta_i sin\theta_o = dot(wi,wo) - mu_i mu_o
	let s: f32 = dot(ωi, ωo) - μi * μo;

	// FON: divide by t = max(mu_i, mu_o) only for s > 0, else t = 1
	let sOverT: f32 = s > 0.0f ? s / max(max(μi, μo), 1e-7f) : s;

	// A_FON(\sigma); B_FON = \sigma A_FON is folded into (1 + \sigma s/t)
	let A: f32 = 1.0f / (1.0f + skFon1 * σ);

	// f_{EON}
	let fOrenNayer: vkof::Spectrum = (
		// \frac{w_d C}{\pi}
		color * (weight * (1.0f / π))
		// A_FON (1 + \sigma \frac{s}{t})
		* (A * (1.0f + σ * sOverT))
	);

	// -- step 2: fujii oren-nayer energy compensation

	// \hat{E}_{ON}(\omega_i)
	let energyCompWi: f32 = (
		bsdfOrenNayerEnergyCompensate(dot(nor, ωi), σ)
	);
	// \hat{E}_{ON}(\omega_o)
	let energyCompWo: f32 = (
		bsdfOrenNayerEnergyCompensate(dot(nor, ωo), σ)
	);

	// \hat{E}_{ON}, this comes from portsmouth 2024
	let albedoAvg: f32 = (
		1.0f / (1.0f + skFon1 * σ) * (1.0f + skFon2 * σ)
	);

	// \rho_{ms}
	let ρMs: vkof::Spectrum = (
		// \frac{C^2}{\pi}
		color*color
		// note the OpenPBR spec implies there should be division by pi here
		// However the original portsmouth impl only has one division, and this
		// passes the furnace test without the additional division.
		// * IPI
		// {\langle \hat{E}_{ON} \rangle / (1 - \langle \hat{E}_{ON} \rangle)}
		* (albedoAvg / max(1.0f - albedoAvg, 1e-6f))
		// / {1 - C(1 - \langle \hat{E}_{ON} \rangle)}
		/ (1.0f - color * (1.0f - albedoAvg))
	);

	// f^{Fujii}_{EON}(\omega_i, \omega_o)
	let fOrenNayerFujii: vkof::Spectrum = (
		// \frac{w_d \rho_{ms}}{\pi}
		ρMs * (weight * (1.0f / π))
		// (1 - \hat{E}_{ON}(\omega_i))
		* max(1e-7f, 1.0f - energyCompWi)
		// (1 - \hat{E}_{ON}(\omega_o))
		* max(1e-7f, 1.0f - energyCompWo)
	);

	// f_{diffuse}(\omega_i, \omega_o)
	let fDiffuse: vkof::Spectrum = (
		// f_{EON}(\omega_i, \omega_o) + f^{Fujii}_{EON}(\omega_i, \omega_o)
		fOrenNayer + fOrenNayerFujii
	);

	return fDiffuse;
}

public func bsdfOpenPbrGlossyDiffuseOrenNayerEvaluateF(
	mat: MaterialBackend_OpenPbr.Params,
	nor: f32v3,
	ωi: f32v3,
	ωo: f32v3,
	λ: Wavelengths,
) -> vkof::Spectrum {
	return (
		bsdfOpenPbrOrenNayerEvaluateF(
			mat.baseColor,
			mat.baseWeight,
			mat.baseDiffuseRoughness,
			nor,
			ωi,
			ωo
		)
	);
}

public func bsdfOpenPbrGlossyDiffusePdf(
	dotNorWo: f32,
) -> f32 {
	// cosine hemisphere; zero below horizon
	return max(dotNorWo, 0.0f) * (1.0f / π);
}

public func bsdfOpenPbrGlossyDiffuseSampleWo(
	nor: f32v3,
	ξ: f32v2,
) -> f32v3 {
	return vkof::cosineHemisphereSampleWo(nor, ξ);
}

// -----------------------------------------------------------------------------
// -- thin-walled subsurface sheet
// -----------------------------------------------------------------------------

/*
	openpbr thin-walled mode: the subsurface slab degenerates into an
	infinitesimally thin sheet of dense scattering material, splitting
	S = subsurface_color between a diffuse reflection and a diffuse
	transmission lobe by g = subsurface_scatter_anisotropy

	f^R_{diffuse} = \frac{1}{2} S (1 - g) f_+ \\
	f^T_{diffuse} = \frac{1}{2} S (1 + g) f_-
		\tag{openpbr eq thin_wall_subsurface}\\
	E_R[f^R_{diffuse}] + E_T[f^T_{diffuse}] = S \le 1\\

	where f_\pm are albedo-1 diffuse lobes in the upper/lower hemisphere,
	both shaped by base_diffuse_roughness. subsurface_radius and
	subsurface_radius_scale are ignored in this mode -- the mean free path
	is infinitesimal
*/
public func bsdfOpenPbrThinWalledSubsurfaceEvaluateF(
	mat: MaterialBackend_OpenPbr.Params,
	nor: f32v3,
	ωi: f32v3,
	ωo: f32v3,
	λ: Wavelengths,
) -> vkof::Spectrum {
	let g: f32 = clamp(mat.subsurfaceScatterAnisotropy, -1.0f, 1.0f);
	let dotNorWo: f32 = dot(nor, ωo);
	// f_-: "oren-nayar lobe flipped into the appropriate hemisphere"
	// keeps μₒ positive
	let ωoLobe: f32v3 = dotNorWo < 0.0f ? ωo - 2.0f * dotNorWo * nor : ωo;
	// f_\pm is albedo 1; S and the split are applied here instead
	let fLobe: vkof::Spectrum = (
		bsdfOpenPbrOrenNayerEvaluateF(
			vkof::Spectrum.one(), 1.0f, mat.baseDiffuseRoughness, nor, ωi, ωoLobe
		)
	);
	// \frac{1}{2}(1 - g) reflecting, \frac{1}{2}(1 + g) transmitting
	let split: f32 = 0.5f * (dotNorWo < 0.0f ? 1.0f + g : 1.0f - g);
	return (
		max(mat.subsurfaceColor, 0.0f)
		* split * fLobe
	);
}

// probability density for thin-walled subsurface sheet
// cosine hemisphere, split by g
public func bsdfOpenPbrThinWalledSubsurfacePdf(
	dotNorWo: f32,
	anisotropy: f32,
) -> f32 {
	let g: f32 = clamp(anisotropy, -1.0f, 1.0f);
	return (
		dotNorWo < 0.0f
		? 0.5f * (1.0f + g) * (-dotNorWo) * (1.0f / π)
		: 0.5f * (1.0f - g) * dotNorWo * (1.0f / π)
	);
}

public func bsdfOpenPbrThinWalledSubsurfaceSampleWo(
	nor: f32v3,
	anisotropy: f32,
	ξ: f32v2,
	uSide: f32,
) -> f32v3 {
	let g: f32 = clamp(anisotropy, -1.0f, 1.0f);
	let transmit: bool = uSide < 0.5f * (1.0f + g);
	return vkof::cosineHemisphereSampleWo(transmit ? -nor : nor, ξ);
}

} // namespace vkof
