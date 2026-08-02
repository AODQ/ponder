module "util-material-openpbr-diffuse";
__exported import "util-types";
__exported import "util-material-openpbr-shared";
__exported import "util-shading-frame";

public func utilOrenNayerEnergyCompensate(
	mu: f32,
	r: f32,
) -> f32 {
	// portsmouth 2024 approximate form;
	// in future can either use a lookup table or a more accurate approximation
	const f32 mucomp = 1.0f - mu;
	const f32 g1 = 0.0571085289f;
	const f32 g2 = 0.491881867f;
	const f32 g3 = -0.332181442f;
	const f32 g4 = 0.0714429953f;
	f32 gOverPi = mucomp * (g1 + mucomp * (g2 + mucomp * (g3 + mucomp * g4)));
	return (1.0f + r * gOverPi) / (1.0f + r * skFon1);
}

// core EON with the albedo passed explicitly rather than read off the
// material, so the thin-walled subsurface sheet below can reuse it as the
// spec's albedo-1 lobe f_\pm
public func openPbrOrenNayerEvaluateF(
	color: f32v3,
	weight: f32,
	sigma: f32,
	nor: f32v3,
	wi: f32v3,
	wo: f32v3,
) -> f32v3 {
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

	// -- step 1: fujii oren-nayer scattering (EON paper, portsmouth 2024);
	// the energy compensation in step 2 recovers (1 - E_FON), so the
	// single-scatter lobe must be the fujii model or the furnace test fails

	const f32 muI = dot(nor, wi);
	const f32 muO = dot(nor, wo);

	// s = cos\phi sin\theta_i sin\theta_o = dot(wi,wo) - mu_i mu_o
	const f32 s = dot(wi, wo) - muI * muO;

	// FON: divide by t = max(mu_i, mu_o) only for s > 0, else t = 1
	const f32 sOverT = s > 0.0f ? s / max(max(muI, muO), 1e-7f) : s;

	// A_FON(\sigma); B_FON = \sigma A_FON is folded into (1 + \sigma s/t)
	const f32 A = 1.0f / (1.0f + skFon1 * sigma);

	// f_{EON}
	const f32v3 fOrenNayer = (
		// \frac{w_d C}{\pi}
		color * weight * IPI
		// A_FON (1 + \sigma \frac{s}{t})
		* (A * (1.0f + sigma * sOverT))
	);

	// -- step 2: fujii oren-nayer energy compensation

	// \hat{E}_{ON}(\omega_i)
	const f32 energyCompWi = (
		utilOrenNayerEnergyCompensate(dot(nor, wi), sigma)
	);
	// \hat{E}_{ON}(\omega_o)
	const f32 energyCompWo = (
		utilOrenNayerEnergyCompensate(dot(nor, wo), sigma)
	);

	// \hat{E}_{ON}, this comes from portsmouth 2024
	const f32 albedoAvg = (
		1.0f / (1.0f + skFon1 * sigma) * (1.0f + skFon2 * sigma)
	);

	// \rho_{ms}
	const f32v3 rhoMs = (
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
	const f32v3 fOrenNayerFujii = (
		// \frac{w_d \rho_{ms}}{\pi}
		weight * rhoMs * IPI
		// (1 - \hat{E}_{ON}(\omega_i))
		* max(1e-7f, 1.0f - energyCompWi)
		// (1 - \hat{E}_{ON}(\omega_o))
		* max(1e-7f, 1.0f - energyCompWo)
	);

	// f_{diffuse}(\omega_i, \omega_o)
	const f32v3 fDiffuse = (
		// f_{EON}(\omega_i, \omega_o) + f^{Fujii}_{EON}(\omega_i, \omega_o)
		fOrenNayer + fOrenNayerFujii
	);

	return fDiffuse;
}

public func openPbrGlossyDiffuseOrenNayerEvaluateF(
	mat: OpenPbrMaterial,
	nor: f32v3,
	wi: f32v3,
	wo: f32v3,
) -> f32v3 {
	return (
		openPbrOrenNayerEvaluateF(
			mat.baseColor, mat.baseWeight, mat.baseDiffuseRoughness, nor, wi, wo
		)
	);
}

public func openPbrGlossyDiffusePdf(
	dotNorWo: f32,
) -> f32 {
	// cosine hemisphere; zero below horizon
	return max(dotNorWo, 0.0f) * IPI;
}

public func openPbrGlossyDiffuseSampleWo(
	nor: f32v3,
	xi: f32v2,
) -> f32v3 {
	return utilCosineHemisphereSampleWo(nor, xi);
}

// -----------------------------------------------------------------------------
// -- thin-walled subsurface sheet
// -----------------------------------------------------------------------------
// lives here rather than in util-material-openpbr-subsurface.glsl (which
// owns the random walk) because these are plain diffuse lobes reusing the
// EON core above

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
public func openPbrThinWalledSubsurfaceEvaluateF(
	mat: OpenPbrMaterial,
	nor: f32v3,
	wi: f32v3,
	wo: f32v3,
) -> f32v3 {
	const f32 g = clamp(mat.subsurfaceScatterAnisotropy, -1.0f, 1.0f);
	const f32 dotNorWo = dot(nor, wo);
	// f_-: the spec's "oren-nayar lobe flipped into the appropriate
	// hemisphere", i.e. the same lobe shape mirrored through the surface
	// plane, which also keeps the EON body's \mu_o positive
	const f32v3 woLobe = dotNorWo < 0.0f ? wo - 2.0f * dotNorWo * nor : wo;
	// f_\pm is albedo 1; S and the split are applied here instead
	const f32v3 fLobe = (
		openPbrOrenNayerEvaluateF(
			f32v3(1.0f), 1.0f, mat.baseDiffuseRoughness, nor, wi, woLobe
		)
	);
	// \frac{1}{2}(1 - g) reflecting, \frac{1}{2}(1 + g) transmitting
	const f32 split = 0.5f * (dotNorWo < 0.0f ? 1.0f + g : 1.0f - g);
	return max(mat.subsurfaceColor, f32v3(0.0f)) * split * fLobe;
}

// hemisphere chosen by the same \frac{1}{2}(1 \mp g) energy split as the
// lobes, cosine-weighted within it
public func openPbrThinWalledSubsurfacePdf(
	dotNorWo: f32,
	anisotropy: f32,
) -> f32 {
	const f32 g = clamp(anisotropy, -1.0f, 1.0f);
	return (
		dotNorWo < 0.0f
		? 0.5f * (1.0f + g) * (-dotNorWo) * IPI
		: 0.5f * (1.0f - g) * dotNorWo * IPI
	);
}

public func openPbrThinWalledSubsurfaceSampleWo(
	nor: f32v3,
	anisotropy: f32,
	xi: f32v2,
	uSide: f32,
) -> f32v3 {
	const f32 g = clamp(anisotropy, -1.0f, 1.0f);
	const bool transmit = uSide < 0.5f * (1.0f + g);
	return utilCosineHemisphereSampleWo(transmit ? -nor : nor, xi);
}

