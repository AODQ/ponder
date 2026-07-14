#ifndef UTIL_MATERIAL_OPENPBR_DIFFUSE_GLSL
#define UTIL_MATERIAL_OPENPBR_DIFFUSE_GLSL

float utilOrenNayerEnergyCompensate(const float mu, const float r) {
	// portsmouth 2024 approximate form;
	// in future can either use a lookup table or a more accurate approximation
	const float mucomp = 1.0f - mu;
	const float g1 = 0.0571085289f;
	const float g2 = 0.491881867f;
	const float g3 = -0.332181442f;
	const float g4 = 0.0714429953f;
	float gOverPi = mucomp * (g1 + mucomp * (g2 + mucomp * (g3 + mucomp * g4)));
	return (1.0f + r * gOverPi) / (1.0f + r * skFon1);
}

vec3 openPbrGlossyDiffuseOrenNayerEvaluateF(
	const OpenPbrMaterial mat,
	const vec3 nor,
	const vec3 wi,
	const vec3 wo
) {
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
	// the energy compensation in step 2 recovers exactly (1 - E_FON), so the
	// single-scatter lobe must be the fujii model or the furnace test fails
	// (see scripts/furnace-test-eon-single-scatter.py)

	// \sigma
	const float sigma = mat.baseDiffuseRoughness;

	const float muI = dot(nor, wi);
	const float muO = dot(nor, wo);

	// s = cos\phi sin\theta_i sin\theta_o = dot(wi,wo) - mu_i mu_o
	const float s = dot(wi, wo) - muI * muO;

	// FON: divide by t = max(mu_i, mu_o) only for s > 0, else t = 1
	const float sOverT = s > 0.0f ? s / max(max(muI, muO), 1e-7f) : s;

	// A_FON(\sigma); B_FON = \sigma A_FON is folded into (1 + \sigma s/t)
	const float A = 1.0f / (1.0f + skFon1 * sigma);

	// f_{EON}
	const vec3 fOrenNayer = (
		// \frac{w_d C}{\pi}
		mat.baseColor * mat.baseWeight * IPI
		// A_FON (1 + \sigma \frac{s}{t})
		* (A * (1.0f + sigma * sOverT))
	);

	// -- step 2: fujii oren-nayer energy compensation

	// \hat{E}_{ON}(\omega_i)
	const float energyCompWi = (
		utilOrenNayerEnergyCompensate(dot(nor, wi), sigma)
	);
	// \hat{E}_{ON}(\omega_o)
	const float energyCompWo = (
		utilOrenNayerEnergyCompensate(dot(nor, wo), sigma)
	);

	// \hat{E}_{ON}, this comes from portsmouth 2024
	const float albedoAvg = (
		1.0f / (1.0f + skFon1 * sigma) * (1.0f + skFon2 * sigma)
	);

	// \rho_{ms}
	const vec3 rhoMs = (
		// \frac{C^2}{\pi}
		mat.baseColor*mat.baseColor
		// note the OpenPBR spec implies there should be division by pi here and
		// that it does not get cancelled. However the original portsmouth impl
		// does cancel the term out, and this passes the furnace test
		// * IPI
		// {\langle \hat{E}_{ON} \rangle / (1 - \langle \hat{E}_{ON} \rangle)}
		* (albedoAvg / max(1.0f - albedoAvg, 1e-6f))
		// / {1 - C(1 - \langle \hat{E}_{ON} \rangle)}
		/ (1.0f - mat.baseColor * (1.0f - albedoAvg))
	);

	// f^{Fujii}_{EON}(\omega_i, \omega_o)
	const vec3 fOrenNayerFujii = (
		// \frac{w_d \rho_{ms}}{\pi}
		mat.baseWeight * rhoMs * IPI
		// (1 - \hat{E}_{ON}(\omega_i))
		* max(1e-7f, 1.0f - energyCompWi)
		// (1 - \hat{E}_{ON}(\omega_o))
		* max(1e-7f, 1.0f - energyCompWo)
	);

	// f_{diffuse}(\omega_i, \omega_o)
	const vec3 fDiffuse = (
		// f_{EON}(\omega_i, \omega_o) + f^{Fujii}_{EON}(\omega_i, \omega_o)
		fOrenNayer + fOrenNayerFujii
	);

	return fDiffuse;
}

float openPbrGlossyDiffusePdf(const float dotNorWo) {
	// cosine hemisphere
	return dotNorWo * IPI;
}

vec3 openPbrGlossyDiffuseSampleWo(
	const vec3 nor,
	inout vec2 seed2
) {
	return utilCosineHemisphereSampleWo(nor, seed2);
}

#endif // UTIL_MATERIAL_OPENPBR_DIFFUSE_GLSL
