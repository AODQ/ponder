// shitty openbr implementation
// https://academysoftwarefoundation.github.io/OpenPBR/index.html

#ifndef UTIL_MATERIAL_OPENPBR_GLSL
#define UTIL_MATERIAL_OPENPBR_GLSL

#include "util-random.glsl"
#include "util-material-openpbr-tables.glsl"
#include "util-nan-probe.glsl"

#define IPI 0.31830988618379067154
#define PI  3.14159265358979323846
#define TAU 6.28318530717958647692
const float skFon1 = 0.5f - 2.0f / (3.0f * PI);
const float skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);
// floors roughness away from the ggx d/v singularity at 0.0
const float skMinRoughness = 1e-3f;

/*
	                                                 ^ emission
	.---------.--------------------------------------|-------.
	|         | fuzz                                 |       |
	|         '--------------------------------------|-------' <- thin film
	|         | clearcoat                            |       |
	|         '-------'--------------'------------'----------'
	|         | metal |              | subsurface | gloss    |
	| ambient '       | translucent  |            |----------'
	| medium  |       | base         |            | diffuse  |
	'---------'-------'--------------'------------'----------'
	                                 <--- opaque base ------->
	                  < ---------- dielectric base ---------->
	          < ------------- base -------------------------->

	the resulting material is denoted M_{pbr}

	M_{pbr} = mix(S_{ambient-medium}, M_{surface}, \alpha)\\
	M_{surface} = mix(M_{coated-base}, S_{fuzz}, F)\\
	M_{coated-base} = mix(M_{base-substrate}, S_{coat}, C)\\
	M_{base-substrate} = mix(M_{dielectric-base}, S_{metal}, M)\\
	M_{dielectric-base} = mix(M_{opaque-base}, S_{translucent-base}, T)\\
	M_{opaque-base} = mix(M_{glossy-diffuse}, S_{subsurface}, S)\\
	M_{glossy-diffuse} = layer(S_{diffuse}, S_{gloss})\\

	where,
		\alpha = geometry opacity,
		F = fuzz weight,
		C = clearcoat weight,
		M = base metallic,
		T = transmission weight,
		S = subsurface weight,
*/

// -----------------------------------------------------------------------------
// -- utility material includes
// -----------------------------------------------------------------------------
// includes that use openpbrmaterial struct and provide material interface

#include "util-material-openpbr-shared.h"

#include "util-material-openpbr-thinfilm.glsl"
#include "util-material-openpbr-microfacet.glsl"

#include "util-material-openpbr-diffuse.glsl"
#include "util-material-openpbr-metallic.glsl"
#include "util-material-openpbr-fuzz.glsl"
#include "util-material-openpbr-coat.glsl"
#include "util-material-openpbr-subsurface.glsl"
#include "util-material-openpbr-transmission.glsl"
#include "util-material-openpbr-metallic.glsl"

// -----------------------------------------------------------------------------
// -- utility material evaluation
// -----------------------------------------------------------------------------


vec3 openPbrBaseSubstrate(
	const OpenPbrMaterial mat
) {
	/*
		M_{bs} = mix( M_{db}, S_{metal}, M)\\
		M_{db} = mix( M_{opaque-base}, S_{translucent-base}, T)\\

		where,
			bs = base substrate
			db = dielectric base
			M = base metallic
			S = specular,
			T = transmission weight
	*/

	const vec3 metallicDielectricBase = vec3(0.0f);//TODO
	const vec3 surfaceMetallic = vec3(0.0f);//TODO
	const vec3 subsurfaceTranslucentBase = vec3(0.0f);//TODO
	const vec3 metallicOpaqueBase = vec3(0.0f);//TODO

	return vec3(0.0f);
}

// -----------------------------------------------------------------------------
// -- public api
// -----------------------------------------------------------------------------

vec3 openPbrEvaluateF(
	const MaterialTableHandles tables,
	const ShadingFrame frame,
	const ShadingFrame coatFrame,
	const vec3 wi,
	const vec3 wo,
	OpenPbrMaterial mat,
	const bool isInsideMedium,
	// this path's hero-wavelength ior; see openPbrTransmissionEvaluateF
	const float dispersedIor
) {
	const vec3 nor = frame.nor;
	// for transmitted wo, reflection lobes produce garbage
	if (dot(nor, wo) < 0.0f) {
		// scope limit, not an oversight: a coat or fuzz layer above a
		// transmissive base is not attenuated here. openpbr figure 96
		// layers coat/fuzz above the whole base substrate including its
		// transmissive component, so a coated/fuzzy glass should lose some
		// energy to the coat's fresnel/absorption on the way through; this
		// falls straight out to the bare transmission lobe instead. no
		// furnace-test coverage exists for coat/fuzz stacked on a
		// transmissive base (only "metal + coat" / "metal + fuzz" today)
		const vec3 fTransmissionOnly = (
			mat.transmissionWeight
			* openPbrTransmissionEvaluateF(
				mat, nor, wi, wo, isInsideMedium, dispersedIor
			)
		);
		NAN_CHECK3(
			fTransmissionOnly,
			"NaN openPbrEvaluateF.fTransmissionOnly px(%d,%d)=%v3f"
		)
		return fTransmissionOnly;
	}
	// relative ior: inverted when reflecting from inside the medium
	const float etaEff = openPbrEffectiveIor(mat.specularIor, mat.specularWeight);
	const float etaRel = isInsideMedium ? (1.0f / etaEff) : etaEff;
	const float iorRatio = (etaEff - 1.0f) / (etaEff + 1.0f);
	const float f0Eff = iorRatio * iorRatio;

	// -- glossy-diffuse base
	const vec3 fDiffuse = (
		openPbrGlossyDiffuseOrenNayerEvaluateF(mat, nor, wi, wo)
	);
	NAN_CHECK3(fDiffuse, "NaN openPbrEvaluateF.fDiffuse px(%d,%d)=%v3f")

	// -- subsurface
	const vec3 fSubsurface = (
		openPbrSubsurfaceEvaluateF(tables, mat, nor, wi, wo)
	);
	NAN_CHECK3(fSubsurface, "NaN openPbrEvaluateF.fSubsurface px(%d,%d)=%v3f")

	// -- translucent specular
	const vec3 fTranslucentSpecular = vec3(0.0f);

	// -- conductor
	const vec3 fConductor = (
		openPbrFresnelMetallicEvaluateF(mat, frame, wi, wo)
	);
	NAN_CHECK3(fConductor, "NaN openPbrEvaluateF.fConductor px(%d,%d)=%v3f")

	// -- specular reflection
	const vec3 fSpecular = (
		openPbrDielectricSpecularEvaluateF(mat, frame, wi, wo, etaRel)
		* mat.specularColor
	);
	NAN_CHECK3(fSpecular, "NaN openPbrEvaluateF.fSpecular px(%d,%d)=%v3f")
	// compensated directional albedo; f0Eff folds in via the fresnel-fit;
	// roughened by a rough coat to match fSpecular's actual lobe width
	const float roughenedSpecularRoughness = (
		openPbrCoatRoughenedRoughness(
			mat.specularRoughness, mat.coatRoughness, mat.coatWeight
		)
	);
	const float fSpecularDirectionalAlbedo = (
		utilMicrofacetDielectricAlbedo(
			dot(nor, wo), roughenedSpecularRoughness, f0Eff
		)
	);
	NAN_CHECK1(
		fSpecularDirectionalAlbedo,
		"NaN openPbrEvaluateF.fSpecularDirectionalAlbedo px(%d,%d)=%f"
	)

	// -- linear combination of lobe components
	const vec3 fTDielectricBase = (
		mix(
			mix(fDiffuse, fSubsurface, mat.subsurfaceWeight),
			fTranslucentSpecular,
			mat.transmissionWeight
		)
	);
	const vec3 fDielectricBase = (
		fSpecular + ((1.0f - fSpecularDirectionalAlbedo) * fTDielectricBase)
	);
	const vec3 fBaseSubstrate = (
		mix(fDielectricBase, fConductor, mat.baseMetalness)
	);

	// -- coat layer over the base substrate; openpbr figure 96
	const vec3 fCoatedBase = (
		openPbrCoatEvaluateF(
			tables, mat, coatFrame, wi, wo, fBaseSubstrate, isInsideMedium
		)
	);
	NAN_CHECK3(fCoatedBase, "NaN openPbrEvaluateF.fCoatedBase px(%d,%d)=%v3f")

	// -- fuzz layer over the coated base; openpbr figure 96
	const vec3 fSurface = (
		openPbrFuzzEvaluateF(tables, mat, fCoatedBase, nor, wi, wo)
	);
	NAN_CHECK3(fSurface, "NaN openPbrEvaluateF.fSurface px(%d,%d)=%v3f")
	return fSurface;
}

// -----------------------------------------------------------------------------
// -- lobe selection
// -----------------------------------------------------------------------------

// which lobe openPbrSampleWo picked, reported via its pickedLobe out param;
// used by callers that need to attribute a sample to a lobe (e.g. the bsdf
// lobe stats test probe) without re-deriving the selection draw themselves
#define OPENPBR_LOBE_COAT 0u
#define OPENPBR_LOBE_SPECULAR 1u
#define OPENPBR_LOBE_DIFFUSE 2u
#define OPENPBR_LOBE_FUZZ 3u
#define OPENPBR_LOBE_TRANSMISSION 4u

// lobe selection probabilities shared by openPbrEvaluatePdf and
// openPbrSampleWo; computed once per (mat, wi) so the two can never disagree
struct OpenPbrLobeSelection {
	float probabilityCoat;
	float probabilitySpecular;
	float probabilityDiffuse;
	float probabilityFuzz;
	float probabilityTransmission;
	float probabilityTotal;
	float fuzzA;
	float fuzzB;
};

OpenPbrLobeSelection openPbrLobeSelection(
	const MaterialTableHandles tables,
	const OpenPbrMaterial mat,
	const float dotNorWi
) {
	// single clamp point for the incident cosine used by all LUT fetches
	const float mu = max(dotNorWi, 1e-5f);
	const float etaEff = openPbrEffectiveIor(mat.specularIor, mat.specularWeight);
	const float iorRatio = (etaEff - 1.0f) / (etaEff + 1.0f);
	const float f0 = iorRatio * iorRatio;

	OpenPbrLobeSelection sel;
	// coat interface albedo with the coat ior and roughness
	const float coatIorRatio = (mat.coatIor - 1.0f) / (mat.coatIor + 1.0f);
	const float coatF0 = coatIorRatio * coatIorRatio;
	sel.probabilityCoat = (
		mat.coatWeight
		* utilMicrofacetDielectricAlbedo(mu, mat.coatRoughness, coatF0)
	);
	NAN_CHECK1(
		sel.probabilityCoat,
		"NaN openPbrLobeSelection.probabilityCoat px(%d,%d)=%f"
	)
	// conductors are all specular: metalness overrides the dielectric
	// interface albedo (the conductor shares the specular ggx lobe, so no
	// separate lobe is needed for sampling); roughened by a rough coat to
	// match the actual evaluated lobe width (openpbr spec, coat roughening)
	const float roughenedSpecularRoughness = (
		openPbrCoatRoughenedRoughness(
			mat.specularRoughness, mat.coatRoughness, mat.coatWeight
		)
	);
	sel.probabilitySpecular = (
		max(
			utilMicrofacetDielectricAlbedo(mu, roughenedSpecularRoughness, f0),
			mat.baseMetalness
		)
	);
	NAN_CHECK1(
		sel.probabilitySpecular,
		"NaN openPbrLobeSelection.probabilitySpecular px(%d,%d)=%f"
	)
	// the substrate below vanishes as metalness rises; what remains splits
	// between the opaque diffuse-like base and the transmissive base
	// according to transmission_weight, mirroring the M_{dielectric-base}
	// = mix(M_{opaque-base}, S_{translucent-base}, T) layering
	const float baseBudget = 1.0f - mat.baseMetalness;
	sel.probabilityDiffuse = baseBudget * (1.0f - mat.transmissionWeight);
	sel.probabilityTransmission = baseBudget * mat.transmissionWeight;
	// zeltner sheen: selection probability from the fitted directional
	// albedo R (LUT .z); a and b ride along for the sample/pdf pair. skip
	// the LUT fetch entirely when there's no fuzz to select in the first
	// place -- probabilityFuzz/fuzzA/fuzzB all come out to exactly the same
	// zero this would have produced anyway
	const vec3 fuzzParams = (
		mat.fuzzWeight > 0.0f
			? utilZeltnerFuzzLookup(tables, mu, mat.fuzzRoughness)
			: vec3(0.0f)
	);
	NAN_CHECK3(fuzzParams, "NaN openPbrLobeSelection.fuzzParams px(%d,%d)=%v3f")
	sel.probabilityFuzz = mat.fuzzWeight * fuzzParams.z;
	sel.probabilityTotal = (
		sel.probabilityCoat + sel.probabilitySpecular
		+ sel.probabilityDiffuse + sel.probabilityFuzz
		+ sel.probabilityTransmission
	);
	NAN_CHECK1(
		sel.probabilityTotal,
		"NaN openPbrLobeSelection.probabilityTotal px(%d,%d)=%f"
	)
	sel.fuzzA = fuzzParams.x;
	sel.fuzzB = fuzzParams.y;

	return sel;

	// const float baseProbability = 

	// const float dielectricProbability = (
	// 	f0
	// 	* utilMicrofacetDielectricEnergyCompensate(
	// 		tables, mu, mat.specularRoughness
	// 	)
	// );
	// const float probabilitySpecular = (
	// 	max(dielectricProbability, mat.baseMetalness)
	// );
	// const float coatIorRatio = (mat.coatIor - 1.0f) / (mat.coatIor + 1.0f);
	// const float coatF0 = coatIorRatio * coatIorRatio;
	// const float probabilityCoat = (
	// 	mat.coatWeight * coatF0
	// 	* utilMicrofacetDielectricEnergyCompensate(
	// 		tables, mu, mat.coatRoughness
	// 	)
	// );
	// const vec3 fuzzParams = (
	// 	utilZeltnerFuzzLookup(tables, mu, mat.fuzzRoughness)
	// );
	// const float probabilityFuzz = mat.fuzzWeight * fuzzParams.z;
	// const float baseBudget = (
	// 	max(0.0f, 1.0f - probabilityCoat - probabilitySpecular - probabilityFuzz)
	// );
	// const float probabilityTransmission = baseBudget * mat.transmissionWeight;
	// const float probabilityDiffuse = baseBudget * (1.0f - mat.transmissionWeight);
	// const float probabilityTotal = (
	// 	probabilityCoat + probabilitySpecular + probabilityDiffuse
	// 	+ probabilityFuzz + probabilityTransmission
	// );
	// return OpenPbrLobeSelection(
	// 	probabilityCoat,
	// 	probabilitySpecular,
	// 	probabilityDiffuse,
	// 	probabilityFuzz,
	// 	probabilityTransmission,
	// 	probabilityTotal,
	// 	fuzzParams.x,
	// 	fuzzParams.y
	// );
}

// mixture pdf over all lobes for a known selection; this is what MIS needs,
// so per-lobe sampling never returns its own pdf
float openPbrEvaluatePdfSelected(
	const OpenPbrLobeSelection sel,
	const OpenPbrMaterial mat,
	const ShadingFrame frame,
	const ShadingFrame coatFrame,
	const vec3 wi,
	const vec3 wo,
	const bool isInsideMedium,
	// must match the dispersedIor openPbrSampleWo/openPbrEvaluateF used
	// for this same wo, or the sampler and pdf disagree
	const float dispersedIor
) {
	const vec3 nor = frame.nor;
	const float dotNorWo = dot(nor, wo);
	// per-axis slopes; must match the sampling in openPbrSampleWo; the
	// coat-roughened base roughness must also match openPbrEvaluateF /
	// openPbrLobeSelection so pdf and evaluated f stay consistent
	const float roughenedSpecularRoughness = (
		openPbrCoatRoughenedRoughness(
			mat.specularRoughness, mat.coatRoughness, mat.coatWeight
		)
	);
	const vec2 specAlpha = (
		openPbrRoughnessAlpha(
			roughenedSpecularRoughness, mat.specularRoughnessAnisotropy
		)
	);
	const vec2 coatAlpha = (
		openPbrRoughnessAlpha(mat.coatRoughness, mat.coatRoughnessAnisotropy)
	);
	// gltf interop anisotropy rotation (not part of openpbr); must match
	// the sampling in openPbrSampleWo. the coat has no rotation source
	const ShadingFrame specFrame = (
		shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
	);

	// transmitted direction: transmission plus the residual below-horizon
	// reflections of the bounded vndf sampler (the eq. 5 cut only removes
	// the band occluded at every azimuth, so at oblique incidence the
	// spec/coat lobes still emit below the horizon at a low rate; f is zero
	// there but the pdf must own the density or sampler and pdf disagree)
	if (dotNorWo <= 0.0f) {
		// thin-walled transmission always samples wo = -wi exactly, so
		// wiLocal + woLocal = 0 there; the leak terms below reconstruct a
		// half vector as normalize(wiLocal + woLocal), which is degenerate
		// at that exact direction and blows up into a huge bogus pdf that
		// swamps the correct transmission term. thin-walled has no bounded
		// vndf reflection sampler to leak from in the first place, so skip
		// the correction entirely
		if (mat.geometryThinWalled > 0.0f) {
			const float pdfThinWalled = (
				sel.probabilityTransmission
				* openPbrTransmissionPdf(
					mat, nor, wi, wo, isInsideMedium, dispersedIor
				)
				/ sel.probabilityTotal
			);
			NAN_CHECK1(
				pdfThinWalled,
				"NaN openPbrEvaluatePdfSelected.pdfThinWalled px(%d,%d)=%f"
			)
			return pdfThinWalled;
		}
		// each term is guarded so a zero-probability lobe contributes exactly
		// 0, rather than trusting probability * pdf to cancel it out: an
		// unselectable lobe's pdf can still be evaluated at a wo miles
		// outside the region it was ever designed for (this wo came from a
		// completely different lobe's sampler) and land on a degenerate
		// (NaN/inf) input; 0 * NaN is NaN, which would poison this whole
		// mixture pdf and every other lobe's throughput along with it
		const float pdfBelowHorizon = (
			(
				(
					sel.probabilityTransmission > 0.0f
					? sel.probabilityTransmission
						* openPbrTransmissionPdf(
							mat, nor, wi, wo, isInsideMedium, dispersedIor
						)
					: 0.0f
				)
				+ (
					sel.probabilitySpecular > 0.0f
					? sel.probabilitySpecular
						* utilMicrofacetGgxBoundedReflectPdfAniso(
							specFrame.tanX, specFrame.tanY, specFrame.nor,
							wi, wo, specAlpha
						)
					: 0.0f
				)
				+ (
					sel.probabilityCoat > 0.0f
					? sel.probabilityCoat
						* utilMicrofacetGgxBoundedReflectPdfAniso(
							coatFrame.tanX, coatFrame.tanY, coatFrame.nor,
							wi, wo, coatAlpha
						)
					: 0.0f
				)
			) / sel.probabilityTotal
		);
		NAN_CHECK1(
			pdfBelowHorizon,
			"NaN openPbrEvaluatePdfSelected.pdfBelowHorizon px(%d,%d)=%f"
		)
		return pdfBelowHorizon;
	}

	// reflection pdfs (bounded vndf, eto & tokuyoshi 2023)
	const float specPdf = (
		utilMicrofacetGgxBoundedReflectPdfAniso(
			specFrame.tanX, specFrame.tanY, specFrame.nor, wi, wo, specAlpha
		)
	);
	NAN_CHECK1(specPdf, "NaN openPbrEvaluatePdfSelected.specPdf px(%d,%d)=%f")
	const float coatPdf = (
		utilMicrofacetGgxBoundedReflectPdfAniso(
			coatFrame.tanX, coatFrame.tanY, coatFrame.nor, wi, wo, coatAlpha
		)
	);
	NAN_CHECK1(coatPdf, "NaN openPbrEvaluatePdfSelected.coatPdf px(%d,%d)=%f")
	const float diffPdf = openPbrGlossyDiffusePdf(dotNorWo);
	NAN_CHECK1(diffPdf, "NaN openPbrEvaluatePdfSelected.diffPdf px(%d,%d)=%f")
	const float fuzzPdf = openPbrFuzzPdf(nor, wi, wo, sel.fuzzA, sel.fuzzB);
	NAN_CHECK1(fuzzPdf, "NaN openPbrEvaluatePdfSelected.fuzzPdf px(%d,%d)=%f")
	// the transmission lobe's vndf sample can also land above the horizon
	// via its total-internal-reflection fallback; its pdf must own that
	// density too or sampler and pdf disagree (see
	// openPbrTransmissionTirReflectPdf)
	const float transmissionTirPdf = (
		openPbrTransmissionTirReflectPdf(
			mat, nor, wi, wo, isInsideMedium, dispersedIor
		)
	);
	NAN_CHECK1(
		transmissionTirPdf,
		"NaN openPbrEvaluatePdfSelected.transmissionTirPdf px(%d,%d)=%f"
	)

	// same zero-probability guarding as the below-horizon branch above: a
	// lobe with no selection weight must contribute exactly 0, not
	// probability(0) * pdf(possibly NaN/inf on an input it was never meant
	// to see)
	const float pdfAboveHorizon = (
		(
			(sel.probabilityCoat > 0.0f ? sel.probabilityCoat * coatPdf : 0.0f)
			+ (
				sel.probabilitySpecular > 0.0f
				? sel.probabilitySpecular * specPdf
				: 0.0f
			)
			+ (
				sel.probabilityDiffuse > 0.0f
				? sel.probabilityDiffuse * diffPdf
				: 0.0f
			)
			+ (sel.probabilityFuzz > 0.0f ? sel.probabilityFuzz * fuzzPdf : 0.0f)
			+ (
				sel.probabilityTransmission > 0.0f
				? sel.probabilityTransmission * transmissionTirPdf
				: 0.0f
			)
		) / sel.probabilityTotal
	);
	NAN_CHECK1(
		pdfAboveHorizon,
		"NaN openPbrEvaluatePdfSelected.pdfAboveHorizon px(%d,%d)=%f"
	)
	return pdfAboveHorizon;
}

float openPbrEvaluatePdf(
	const MaterialTableHandles tables,
	const OpenPbrMaterial mat,
	const ShadingFrame frame,
	const ShadingFrame coatFrame,
	const vec3 wi,
	const vec3 wo,
	const bool isInsideMedium,
	// this path's hero-wavelength ior; see openPbrTransmissionEvaluateF
	const float dispersedIor
) {
	const OpenPbrLobeSelection sel = (
		openPbrLobeSelection(tables, mat, dot(frame.nor, wi))
	);
	return (
		openPbrEvaluatePdfSelected(
			sel, mat, frame, coatFrame, wi, wo, isInsideMedium, dispersedIor
		)
	);
}

vec3 openPbrSampleWo(
	const MaterialTableHandles tables,
	const ShadingFrame frame,
	const ShadingFrame coatFrame,
	const vec3 wi,
	const OpenPbrMaterial mat,
	out float pdf,
	out bool sampledTransmission,
	// which lobe was drawn (OPENPBR_LOBE_*); distinct from
	// sampledTransmission, which is false on a total-internal-reflection
	// fallback even though the transmission lobe was the one picked
	out uint pickedLobe,
	// pcg32 stream (util-random.glsl); the lobe-selection scalar and the
	// picked lobe's directional pair are sequential draws from this one
	// state
	inout u64 state,
	const bool isInsideMedium,
	// this path's hero-wavelength ior; see openPbrTransmissionEvaluateF
	const float dispersedIor
) {
	const vec3 nor = frame.nor;
	// walk through the layers of materials picking sample
	const OpenPbrLobeSelection sel = (
		openPbrLobeSelection(tables, mat, dot(nor, wi))
	);
	const float u = fnSampleUniform(state);

	const bool isCoat = (
		u < sel.probabilityCoat / sel.probabilityTotal
	);
	const bool isSpecular = (
		u < (
			(sel.probabilityCoat + sel.probabilitySpecular)
			/ sel.probabilityTotal
		)
	);
	const bool isFuzz = (
		u < (
			(
				sel.probabilityCoat
				+ sel.probabilitySpecular
				+ sel.probabilityFuzz
			) / sel.probabilityTotal
		)
	);
	const bool isTransmission = (
		u < (
			(
				sel.probabilityCoat
				+ sel.probabilitySpecular
				+ sel.probabilityFuzz
				+ sel.probabilityTransmission
			) / sel.probabilityTotal
		)
	);

	vec3 wo;
	sampledTransmission = false;
	if (isCoat) {
		pickedLobe = OPENPBR_LOBE_COAT;
		const vec2 xi = fnSampleUniform2(state);
		// bounded vndf (eto & tokuyoshi 2023); per-axis slopes must match
		// openPbrEvaluatePdfSelected (the mapping floors at 1e-5)
		const vec2 coatAlpha = (
			openPbrRoughnessAlpha(
				mat.coatRoughness, mat.coatRoughnessAnisotropy
			)
		);
		wo = (
			utilMicrofacetSampleGgxBoundedWoAniso(
				coatFrame.tanX, coatFrame.tanY, coatFrame.nor, wi, coatAlpha, xi
			)
		);
	} else if (isSpecular) {
		pickedLobe = OPENPBR_LOBE_SPECULAR;
		const vec2 xi = fnSampleUniform2(state);
		// coat-roughened base roughness; must match openPbrEvaluatePdfSelected
		const float roughenedSpecularRoughness = (
			openPbrCoatRoughenedRoughness(
				mat.specularRoughness, mat.coatRoughness, mat.coatWeight
			)
		);
		const vec2 specAlpha = (
			openPbrRoughnessAlpha(
				roughenedSpecularRoughness, mat.specularRoughnessAnisotropy
			)
		);
		// gltf interop anisotropy rotation; must match
		// openPbrEvaluatePdfSelected
		const ShadingFrame specFrame = (
			shadingFrameRotate(
				frame, mat.specularRoughnessAnisotropyRotation
			)
		);
		wo = (
			utilMicrofacetSampleGgxBoundedWoAniso(
				specFrame.tanX, specFrame.tanY, specFrame.nor,
				wi, specAlpha, xi
			)
		);
	} else if (isFuzz) {
		pickedLobe = OPENPBR_LOBE_FUZZ;
		const vec2 xi = fnSampleUniform2(state);
		wo = openPbrFuzzSampleWo(nor, wi, sel.fuzzA, sel.fuzzB, xi);
	} else if (isTransmission) {
		pickedLobe = OPENPBR_LOBE_TRANSMISSION;
		if (mat.geometryThinWalled > 0.0f) {
			// infinitely thin shell (khr_materials_transmission without
			// khr_materials_volume): no second surface to refract back out
			// of, so light passes straight through undeviated instead of
			// bending like a solid dielectric volume
			wo = -wi;
			sampledTransmission = true;
		} else {
			const vec2 xi = fnSampleUniform2(state);
			const float roughness = max(mat.specularRoughness, 1e-5f);
			const vec3 h = utilMicrofacetSampleGgxVndf(nor, wi, roughness, xi);
			// etaI/etaT is n_incident/n_transmitted for GLSL refract;
			// dispersedIor replaces the flat mat.specularIor so the bend
			// angle follows this path's locked hero wavelength (see
			// openPbrDispersionIorRgb / utilIrradiance)
			const float etaI = isInsideMedium ? dispersedIor : 1.0f;
			const float etaT = isInsideMedium ? 1.0f : dispersedIor;
			const vec3 refracted = refract(-wi, h, etaI / etaT);
			if (length(refracted) > 0.5f) {
				wo = refracted;
				sampledTransmission = true;
			} else {
				// total internal reflection; the reflection-side pdf owns
				// this density via openPbrTransmissionTirReflectPdf
				wo = normalize(2.0f * dot(wi, h) * h - wi);
			}
		}
	} else {
		pickedLobe = OPENPBR_LOBE_DIFFUSE;
		const vec2 xi = fnSampleUniform2(state);
		wo = openPbrGlossyDiffuseSampleWo(nor, xi);
	}
	// pickedLobe pins down which of the branches above produced wo, since
	// the OPENPBR_LOBE_* value rides along in the printf itself; not a
	// NAN_CHECK3 call since that macro has no slot for the extra lobe arg
#ifdef PONDER_NAN_PROBES
	if (any(isnan(wo)) || any(isinf(wo))) {
		if (nanProbeReserveSlot()) {
			debugPrintfEXT(
				"NaN openPbrSampleWo.wo px(%d,%d) lobe=%u wo=%v3f",
				gNanProbeCoord.x, gNanProbeCoord.y, pickedLobe, wo
			);
		}
	}
#endif
	pdf = (
		openPbrEvaluatePdfSelected(
			sel, mat, frame, coatFrame, wi, wo, isInsideMedium, dispersedIor
		)
	);
	NAN_CHECK1(pdf, "NaN openPbrSampleWo.pdf px(%d,%d)=%f")
	return wo;
}

#endif // UTIL_MATERIAL_OPENPBR_GLSL
