module "util-material-openpbr-ibl";
__exported import "util-types";
__exported import "util-material-openpbr-tables";
__exported import "util-material-openpbr-microfacet";
__exported import "util-material-openpbr-fuzz";


/*
	raster-friendly (image-based lighting / split-sum) counterparts of the
	path-traced openPbrEvaluateF lobe stack (util-material-openpbr*.glsl).
	a raster resolve pass has no per-pixel BSDF sampling budget, so instead
	of evaluating f(wi, wo) at a stochastically sampled wo, IBL integrates
	the whole hemisphere via prefiltered environment maps under the
	standard split-sum assumption N = V = R (karis 2013, "real shading in
	unreal engine 4"): the view/mirror-reflection direction shares wi's
	cosine to the normal, so every wo-cosine term in the ported eval
	functions collapses to the same dotNorWi used for wi.

	composition mirrors openPbrEvaluateF's layering order exactly (coat
	over the metalness-mixed conductor/dielectric base, fuzz over all of
	it). thin-film only reaches the conductor lobe here, matching the
	dielectric specular lobe's own scope (util-dielectric-specular.glsl,
	which is unused dead code -- the live dielectric specular eval is in
	util-material-openpbr-microfacet.glsl and does support thin-film,
	unlike this IBL path). out of scope for this pass: subsurface,
	transmission -- neither has a raster-friendly treatment yet; see
	openPbrIblEvaluateF.
*/

// exact closed form of openPbrFresnelMetallicEvaluateF's directional
// albedo. F82(dotNorWi) and the turquin energy-compensation multiplier
// both depend on wi only, not wo, so they pull straight out of the
// hemispherical integral, leaving utilMicrofacetGgxDirectionalAlbedo(mu,
// alpha, 1, 1) -- exactly "ess" below, the same building block
// openPbrFresnelMetallicEvaluateF's own energyCompensation term uses -- as
// the only wo-dependent piece
/*
	E(\mu_i) = F_{82}(\mu_i)
		\bigl(E_{ss}(\mu_i, \alpha) + F_{avg} (1 - E_{ss}(\mu_i, \alpha))\bigr)
*/
public func openPbrConductorIblAlbedo(
	mat: OpenPbrMaterial,
	dotNorWi: f32,
	roughenedSpecularRoughness: f32,
) -> f32v3 {
	if (mat.baseMetalness <= 0.0f) {
		return f32v3(0.0f);
	}
	const f32 mu = max(dotNorWi, 1e-5f);

	// f82-tint fresnel at mu, ported straight from
	// openPbrFresnelMetallicEvaluateF (its own fresnel term is already
	// mu-only, no half-vector dependence)
	const f32v3 f0 = mat.baseWeight * mat.baseColor;
	const f32v3 fresnelConstS = f0 + (f32v3(1.0f) - f0) * 0.462664f;
	const f32v3 fresnelConst = (
		fresnelConstS * (f32v3(1.0f) - mat.specularWeight * mat.specularColor)
	);
	const f32 fresnelDenom = 0.056653f;
	const f32v3 fs = f0 + (f32v3(1.0f) - f0) * pow(1.0f - mu, 5.0f);
	const f32v3 f82 = (
		max(
			fs - ((mu * pow(1.0f - mu, 6.0f) / fresnelDenom) * fresnelConst),
			f32v3(0.0f)
		)
	);

	// thin-film interference: the real eval mixes this in via
	// dot(h, wi), which varies across the lobe footprint; under N=V=R
	// this collapses to mu (all(h == nor)), so this is exact at the sampled
	// mirror direction but not across the lobe's full angular spread --
	// a common real-time approximation for thin-film IBL, not a literal
	// hemisphere-integrated albedo of the interference term itself
	const f32v3 mfFresnel = (
		mat.thinFilmWeight > 0.0f
		? lerp(
			f82,
			openPbrThinfilmFresnel(mat, mu, openPbrIorFromF0(f0), f32v3(0.0f)),
			mat.thinFilmWeight
		)
		: f82
	);

	const f32v2 alpha = (
		openPbrRoughnessAlpha(
			roughenedSpecularRoughness, mat.specularRoughnessAnisotropy
		)
	);
	const f32v3 fAvg = f0 + (f32v3(1.0f) - f0) * (1.0f / 21.0f);
	const f32 ess = (
		max(
			utilMicrofacetGgxDirectionalAlbedo(
				mu, sqrt(alpha.x * alpha.y), 1.0f, 1.0f
			),
			1e-4f
		)
	);
	return mfFresnel * (ess + fAvg * (1.0f - ess));
}

public struct OpenPbrCoatIblResult {
	public f32v3 ownReflectance;
	public f32v3 attenuation;
};

// coat's own reflectance + attenuation of what's below, ported from
// openPbrCoatEvaluateF under the N = V = R split-sum assumption: wo is
// exactly wi's mirror reflection about the coat normal, so woLocal.z ==
// wiLocal.z == dotNorWi and the half vector all(h == nor) (dotHWi == dotNorWi
// too), collapsing every wi/wo pair in the reference eval to one mu
public func openPbrCoatIblEvaluate(
	mat: OpenPbrMaterial,
	dotNorWi: f32,
	coatPrefilteredSpecular: f32v3,
	isInsideMedium: bool,
) -> OpenPbrCoatIblResult {
	const f32 iorRatio = (mat.coatIor - 1.0f) / (mat.coatIor + 1.0f);
	const f32 f0 = iorRatio * iorRatio;
	if (f0 == 0.0f || mat.coatWeight <= 0.0f) {
		OpenPbrCoatIblResult none;
		none.ownReflectance = f32v3(0.0f);
		none.attenuation = f32v3(1.0f);
		return none;
	}

	const f32 mu = max(dotNorWi, 0.0f);
	const f32 coatEtaRel = isInsideMedium ? (1.0f / mat.coatIor) : mat.coatIor;

	// coat's own directional albedo at mu -- the same closed form
	// openPbrCoatEvaluateF uses for its transmittance term and
	// openPbrLobeSelection for probabilityCoat
	const f32 coatAlbedo = (
		utilMicrofacetDielectricAlbedo(mu, mat.coatRoughness, f0)
	);

	// eq. 72 view-dependent absorption exponent; muIt == muOt under N=V=R,
	// so eq. 71's 0.5*(1/muIt + 1/muOt) collapses to 1/muT
	const f32 muTSq = (
		max(1.0f - (1.0f - mu * mu) / (coatEtaRel * coatEtaRel), 1e-4f)
	);
	const f32v3 coatAbsorption = (
		pow(clamp(mat.coatColor, 1e-6f, 1.0f), f32v3(1.0f / sqrt(muTSq)))
	);

	// eq. 61-70 coat darkening; kSmooth/kRough/baseFresnelWo are all
	// evaluated at wo's cosine, which under N=V=R is just mu
	const f32 kSmooth = utilMicrofacetFresnelDielectric(mu, coatEtaRel);
	const f32 coatFresnelAvg = f0 + (1.0f - f0) * (1.0f / 21.0f);
	const f32 kRough = (
		clamp(
			1.0f - (1.0f - coatFresnelAvg) / (coatEtaRel * coatEtaRel), 0.0f, 1.0f
		)
	);
	const f32 baseFresnelWo = (
		utilMicrofacetFresnelDielectric(mu, mat.specularIor)
	);
	const f32 roughnessDielectric = (
		lerp(
			1.0f, mat.specularRoughness,
			clamp(mat.specularWeight * baseFresnelWo, 0.0f, 1.0f)
		)
	);
	const f32 roughnessBase = (
		lerp(roughnessDielectric, mat.specularRoughness, mat.baseMetalness)
	);
	const f32 kCoat = min(lerp(kSmooth, kRough, roughnessBase), 0.9999f);
	const f32v3 albedoOpaqueBase = (
		lerp(
			mat.baseWeight * mat.baseColor, mat.subsurfaceColor, mat.subsurfaceWeight
		)
	);
	const f32v3 albedoDielectricBase = (
		lerp(albedoOpaqueBase, mat.transmissionColor, mat.transmissionWeight)
	);
	const f32v3 albedoBase = (
		clamp(
			lerp(
				albedoDielectricBase, mat.baseWeight * mat.baseColor, mat.baseMetalness
			),
			0.0f,
			1.0f
		)
	);
	const f32v3 darkeningDelta = (1.0f - kCoat) / (1.0f - albedoBase * kCoat);
	const f32v3 darkening = (
		lerp(f32v3(1.0f), darkeningDelta, mat.coatWeight * mat.coatDarkening)
	);

	OpenPbrCoatIblResult result;
	result.ownReflectance = (
		mat.coatWeight * coatAlbedo * coatPrefilteredSpecular
	);
	// (TODO REVIEW)
	// single (1 - coatAlbedo), matching openPbrCoatEvaluateF's one-sided
	// albedo-scaling; under N=V=R there is only one mu to evaluate it at
	result.attenuation = (
		darkening
		* lerp(
			f32v3(1.0f),
			(1.0f - coatAlbedo) * coatAbsorption,
			mat.coatWeight
		)
	);
	// (TODO REVIEW)
	return result;
}

public struct OpenPbrFuzzIblResult {
	public f32v3 ownReflectance;
	public f32 attenuation;
};

// fuzz's attenuation of what's below is exact (the same zeltner directional
// albedo R the reference eval and lobe selection already use); fuzz's own
// reflectance is NOT a literal port -- a proper prefiltered LTC sheen probe
// is future work, so its own energy is approximated against the diffuse-
// irradiance probe, the same simplification production sheen-IBL
// implementations use for this broad, near-diffuse lobe
public func openPbrFuzzIblEvaluate(
	tables: MaterialTableHandles,
	mat: OpenPbrMaterial,
	dotNorWi: f32,
	irradiance: f32v3,
) -> OpenPbrFuzzIblResult {
	if (mat.fuzzWeight <= 0.0f) {
		OpenPbrFuzzIblResult none;
		none.ownReflectance = f32v3(0.0f);
		none.attenuation = 1.0f;
		return none;
	}
	const f32 eFuzz = (
		utilZeltnerFuzzLookup(tables, max(dotNorWi, 0.0f), mat.fuzzRoughness).z
	);
	OpenPbrFuzzIblResult result;
	result.ownReflectance = mat.fuzzWeight * eFuzz * mat.fuzzColor * irradiance;
	result.attenuation = 1.0f - mat.fuzzWeight * eFuzz;
	return result;
}

// top-level IBL evaluate; mirrors openPbrEvaluateF's composition order
// exactly (dielectric-specular+diffuse mixed by metalness with the
// conductor, then coat, then fuzz). unlike openPbrEvaluateF, this takes
// already-resolved environment samples rather than doing its own texture
// lookups -- irradiance and both prefiltered specular samples (base
// roughness and coat roughness mips, each at its own N=V=R reflect
// direction) are the caller's job, since only the caller has the env map
// handles/mip count
public func openPbrIblEvaluateF(
	tables: MaterialTableHandles,
	mat: OpenPbrMaterial,
	dotNorWi: f32,
	dotCoatNorWi: f32,
	irradiance: f32v3,
	prefilteredSpecularBase: f32v3,
	prefilteredSpecularCoat: f32v3,
	isInsideMedium: bool,
) -> f32v3 {
	const f32 mu = max(dotNorWi, 1e-5f);
	const f32 etaEff = openPbrSpecularEffectiveIor(mat);
	const f32 iorRatio = (etaEff - 1.0f) / (etaEff + 1.0f);
	const f32 f0Eff = iorRatio * iorRatio;
	const f32 roughenedSpecularRoughness = (
		openPbrCoatRoughenedRoughness(
			mat.specularRoughness, mat.coatRoughness, mat.coatWeight
		)
	);

	// gated by specularWeight: the rational fit hardcodes f90=1 (a real
	// interface's grazing fresnel), so it doesn't vanish as f0Eff->0 the
	// way the real per-angle fresnel does at eta=1 -- ungated, a
	// specularWeight=0 material would still darken/reflect
	const f32 fSpecularAlbedo = (
		mat.specularWeight
		* utilMicrofacetDielectricAlbedo(mu, roughenedSpecularRoughness, f0Eff)
	);
	const f32v3 specularOwn = (
		fSpecularAlbedo * mat.specularColor * prefilteredSpecularBase
	);
	// EON diffuse simplified to lambertian for IBL: exact at
	// baseDiffuseRoughness == 0, an approximation otherwise -- a full EON
	// directional-albedo IBL treatment for sigma > 0 is unported
	const f32v3 diffuseIbl = mat.baseColor * mat.baseWeight * irradiance;
	const f32v3 dielectricBaseIbl = (
		specularOwn + (1.0f - fSpecularAlbedo) * diffuseIbl
	);

	const f32v3 conductorOwn = (
		openPbrConductorIblAlbedo(mat, mu, roughenedSpecularRoughness)
		* prefilteredSpecularBase
	);
	const f32v3 baseSubstrateIbl = (
		lerp(dielectricBaseIbl, conductorOwn, mat.baseMetalness)
	);

	const OpenPbrCoatIblResult coat = (
		openPbrCoatIblEvaluate(
			mat, dotCoatNorWi, prefilteredSpecularCoat, isInsideMedium
		)
	);
	const f32v3 coatedIbl = (
		coat.ownReflectance + coat.attenuation * baseSubstrateIbl
	);

	const OpenPbrFuzzIblResult fuzz = (
		openPbrFuzzIblEvaluate(tables, mat, mu, irradiance)
	);
	return fuzz.ownReflectance + fuzz.attenuation * coatedIbl;
}

