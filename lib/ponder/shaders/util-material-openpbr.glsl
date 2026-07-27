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

// evaluate all interfaces of openpbr material
// openpbr eq 96
vec3 openPbrEvaluateF(
	const MaterialTableHandles tables,
	const ShadingFrame frame,
	const ShadingFrame coatFrame,
	const vec3 wi,
	const vec3 wo,
	OpenPbrMaterial mat,
	const bool isInsideMedium,
	const float dispersedIor
) {
	const vec3 nor = frame.nor;

	// for transmitted wo, only do transmission and (in thin-walled mode) the
	// subsurface sheet's diffuse transmission lobe
	if (dot(nor, wo) < 0.0f) {
		vec3 fTransmitted = (
			mat.transmissionWeight
			* openPbrTransmissionEvaluateF(
				mat, frame, wi, wo, isInsideMedium, dispersedIor
			)
		);
		if (mat.geometryThinWalled > 0.0f && mat.subsurfaceWeight > 0.0f) {
			// the sheet occupies the opaque-base slot, so it carries the same
			// (1 - E_spec)(1 - T)(1 - M) weights fSubsurface gets below; coat
			// and fuzz are ignored on the underside per the spec's own
			// thin-walled approximation
			const float etaEffThin = (
				openPbrEffectiveIor(mat.specularIor, mat.specularWeight)
			);
			const float iorRatioThin = (
				(etaEffThin - 1.0f) / (etaEffThin + 1.0f)
			);
			const float essThin = (
				utilMicrofacetDielectricAlbedo(
					abs(dot(nor, wo)),
					openPbrCoatRoughenedRoughness(
						mat.specularRoughness, mat.coatRoughness, mat.coatWeight
					),
					iorRatioThin * iorRatioThin
				)
			);
			fTransmitted += (
				(1.0f - essThin)
				* (1.0f - mat.transmissionWeight)
				* mat.subsurfaceWeight
				* (1.0f - mat.baseMetalness)
				* openPbrThinWalledSubsurfaceEvaluateF(mat, nor, wi, wo)
			);
		}
		NAN_CHECK3(
			fTransmitted,
			"NaN openPbrEvaluateF.fTransmitted px(%d,%d)=%v3f"
		)
		return fTransmitted;
	}

	// -- relative ior
	const float etaEff = openPbrEffectiveIor(mat.specularIor, mat.specularWeight);
	const float etaRel = isInsideMedium ? (1.0f / etaEff) : etaEff;
	const float iorRatio = (etaEff - 1.0f) / (etaEff + 1.0f);
	const float f0Eff = iorRatio * iorRatio;

	// -- glossy-diffuse base
	const vec3 fDiffuse = (
		openPbrGlossyDiffuseOrenNayerEvaluateF(mat, nor, wi, wo)
	);
	NAN_CHECK3(fDiffuse, "NaN openPbrEvaluateF.fDiffuse px(%d,%d)=%v3f")

	// -- subsurface; thin-walled swaps the walk for the diffuse sheet, which
	// unlike the walk does have an instantaneous value on this side
	const vec3 fSubsurface = (
		mat.geometryThinWalled > 0.0f
		? openPbrThinWalledSubsurfaceEvaluateF(mat, nor, wi, wo)
		: openPbrSubsurfaceEvaluateF(tables, mat, nor, wi, wo)
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

	// -- compensated directional albedo
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

	// -- combination of lobe components
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

// which lobe openPbrSampleWo picked

#define OPENPBR_LOBE_COAT 0u
#define OPENPBR_LOBE_SPECULAR 1u
#define OPENPBR_LOBE_DIFFUSE 2u
#define OPENPBR_LOBE_FUZZ 3u
#define OPENPBR_LOBE_TRANSMISSION 4u
#define OPENPBR_LOBE_SUBSURFACE_ENTRY 5u

// lobe selection probabilities
struct OpenPbrLobeSelection {
	float probabilityCoat;
	float probabilitySpecular;
	float probabilityDiffuse;
	float probabilityFuzz;
	float probabilityTransmission;
	float probabilitySubsurfaceEntry;
	float probabilityTotal;
	float fuzzA;
	float fuzzB;
};

// selects a lobe to sample from, returning the selection probabilities and
// zeltner fuzz lobe parameters for the selected \omega_i
OpenPbrLobeSelection openPbrLobeSelection(
	const MaterialTableHandles tables,
	const OpenPbrMaterial mat,
	const float dotNorWi
) {
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

	// metalness overrides the dielectric interface albedo 
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
	const float baseBudget = 1.0f - mat.baseMetalness;
	const float opaqueBudget = baseBudget * (1.0f - mat.transmissionWeight);
	sel.probabilityDiffuse = opaqueBudget * (1.0f - mat.subsurfaceWeight);
	sel.probabilitySubsurfaceEntry = opaqueBudget * mat.subsurfaceWeight;
	sel.probabilityTransmission = baseBudget * mat.transmissionWeight;
	// zeltner sheen; selection probability from the fitted directional R(\mu)
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
		+ sel.probabilityTransmission + sel.probabilitySubsurfaceEntry
	);
	NAN_CHECK1(
		sel.probabilityTotal,
		"NaN openPbrLobeSelection.probabilityTotal px(%d,%d)=%f"
	)
	sel.fuzzA = fuzzParams.x;
	sel.fuzzB = fuzzParams.y;

	return sel;
}

// the subsurface budget buys the lower-hemisphere entry lobe by default and
// the two-sided thin-walled sheet in thin-walled mode; both are zero above
// the horizon in the default mode, so callers can add this unconditionally
float openPbrSubsurfaceSlotPdf(
	const OpenPbrMaterial mat,
	const float dotNorWo
) {
	return (
		mat.geometryThinWalled > 0.0f
		? openPbrThinWalledSubsurfacePdf(
			dotNorWo, mat.subsurfaceScatterAnisotropy
		)
		: openPbrSubsurfaceEntryPdf(dotNorWo)
	);
}

// mixture pdf over all lobes selection
float openPbrEvaluatePdfSelected(
	const OpenPbrLobeSelection sel,
	const OpenPbrMaterial mat,
	const ShadingFrame frame,
	const ShadingFrame coatFrame,
	const vec3 wi,
	const vec3 wo,
	const bool isInsideMedium,
	const float dispersedIor
) {
	const vec3 nor = frame.nor;
	const float dotNorWo = dot(nor, wo);
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
	const ShadingFrame specFrame = (
		shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
	);

	if (dotNorWo <= 0.0f) {
		if (mat.geometryThinWalled > 0.0f) {
			const float pdfThinWalled = (
				(
					sel.probabilityTransmission
					* openPbrTransmissionPdf(
						mat, frame, wi, wo, isInsideMedium, dispersedIor
					)
					+ (
						sel.probabilitySubsurfaceEntry > 0.0f
						? sel.probabilitySubsurfaceEntry
							* openPbrSubsurfaceSlotPdf(mat, dotNorWo)
						: 0.0f
					)
				) / sel.probabilityTotal
			);
			NAN_CHECK1(
				pdfThinWalled,
				"NaN openPbrEvaluatePdfSelected.pdfThinWalled px(%d,%d)=%f"
			)
			return pdfThinWalled;
		}
		const float pdfBelowHorizon = (
			(
				(
					sel.probabilityTransmission > 0.0f
					? sel.probabilityTransmission
						* openPbrTransmissionPdf(
							mat, frame, wi, wo, isInsideMedium, dispersedIor
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
				+ (
					sel.probabilitySubsurfaceEntry > 0.0f
					? sel.probabilitySubsurfaceEntry
						* openPbrSubsurfaceSlotPdf(mat, dotNorWo)
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
	const float transmissionTirPdf = (
		openPbrTransmissionTirReflectPdf(
			mat, frame, wi, wo, isInsideMedium, dispersedIor
		)
	);
	NAN_CHECK1(
		transmissionTirPdf,
		"NaN openPbrEvaluatePdfSelected.transmissionTirPdf px(%d,%d)=%f"
	)

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
			// zero unless thin-walled: the entry lobe is lower-hemisphere only
			+ (
				sel.probabilitySubsurfaceEntry > 0.0f
				? sel.probabilitySubsurfaceEntry
					* openPbrSubsurfaceSlotPdf(mat, dotNorWo)
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
	// true if subsurface entry lobe was picked
	out bool sampledSubsurfaceEntry,
	// which lobe was drawn (OPENPBR_LOBE_*)
	out uint pickedLobe,
	inout u64 state,
	const bool isInsideMedium,
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
	const bool isSubsurfaceEntry = (
		u < (
			(
				sel.probabilityCoat
				+ sel.probabilitySpecular
				+ sel.probabilityFuzz
				+ sel.probabilityTransmission
				+ sel.probabilitySubsurfaceEntry
			) / sel.probabilityTotal
		)
	);

	vec3 wo;
	sampledTransmission = false;
	sampledSubsurfaceEntry = false;
	if (isCoat) {
		pickedLobe = OPENPBR_LOBE_COAT;
		const vec2 xi = fnSampleUniform2(state);
		// bounded vndf (eto & tokuyoshi 2023); must match
		// openPbrEvaluatePdfSelected
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
			// thin shell (khr_materials_transmission without
			// khr_materials_volume); wo = -wo
			wo = -wi;
			sampledTransmission = true;
		} else {
			const vec2 xi = fnSampleUniform2(state);
			// unbounded vndf (the bounded sampler's cap is reflection-only);
			// slopes and frame must match openPbrTransmissionPdf
			const vec2 transmissionAlpha = (
				openPbrRoughnessAlpha(
					mat.specularRoughness, mat.specularRoughnessAnisotropy
				)
			);
			const ShadingFrame transmissionFrame = (
				shadingFrameRotate(
					frame, mat.specularRoughnessAnisotropyRotation
				)
			);
			const vec3 h = (
				utilMicrofacetSampleGgxVndfAniso(
					transmissionFrame.tanX, transmissionFrame.tanY,
					transmissionFrame.nor, wi, transmissionAlpha, xi
				)
			);
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
	} else if (isSubsurfaceEntry) {
		pickedLobe = OPENPBR_LOBE_SUBSURFACE_ENTRY;
		const vec2 xi = fnSampleUniform2(state);
		if (mat.geometryThinWalled > 0.0f) {
			// a thin sheet has no interior to walk through; this is an
			// ordinary two-sided diffuse bsdf, so sampledSubsurfaceEntry
			// stays false and the caller applies the usual f*cos/pdf
			wo = (
				openPbrThinWalledSubsurfaceSampleWo(
					nor, mat.subsurfaceScatterAnisotropy, xi,
					fnSampleUniform(state)
				)
			);
		} else {
			wo = openPbrSubsurfaceEntrySampleWo(nor, xi);
			sampledSubsurfaceEntry = true;
		}
	} else {
		pickedLobe = OPENPBR_LOBE_DIFFUSE;
		const vec2 xi = fnSampleUniform2(state);
		wo = openPbrGlossyDiffuseSampleWo(nor, xi);
	}
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
