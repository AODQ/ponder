implementing bsdf;

import types;
import util;
import material;

#include "include/inline-expression.slang"

// directions in the free functions point away from the surface: ωi there is
// the path's -ωi (see units.slang); Bsdf_OpenPbr negates it once at entry

namespace vkof {

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

	M_{pbr} = mix(S_{ambient-medium}, M_{surface}, \alpha)\\
	M_{surface} = mix(M_{coated-base}, S_{fuzz}, F)\\
	M_{coated-base} = mix(M_{base-substrate}, S_{coat}, C)\\
	M_{base-substrate} = mix(M_{dielectric-base}, S_{metal}, M)\\
	M_{dielectric-base} = mix(M_{opaque-base}, S_{translucent-base}, T)\\
	M_{opaque-base} = mix(M_{glossy-diffuse}, S_{subsurface}, S)\\
	M_{glossy-diffuse} = layer(S_{diffuse}, S_{gloss})\\
*/

// evaluate all interfaces of openpbr material
// openpbr eq 96
public func bsdfOpenPbrEvaluateF(
	zeltnerLtcParamHandle: u32,
	frame: ShadingFrame,
	coatFrame: ShadingFrame,
	ωi: f32v3,
	ωo: f32v3,
	mat: MaterialBackend_OpenPbr.Params,
	isInsideMedium: bool,
	dispersedIor: f32,
	λ: Wavelengths,
) -> vkof::Spectrum {
	let nor: f32v3 = frame.nor;

	// for transmitted wo, only do transmission and (in thin-walled mode) the
	// subsurface sheet's diffuse transmission lobe
	if (dot(nor, ωo) < 0.0f) {
		var fTransmitted: vkof::Spectrum = (
			mat.transmissionWeight
			* bsdfOpenPbrTransmissionEvaluateF(
				mat, frame, ωi, ωo, isInsideMedium, dispersedIor, λ
			)
		);
		if (mat.geometryThinWalled > 0.0f && mat.subsurfaceWeight > 0.0f) {
			// the sheet occupies the opaque-base slot, so it carries the same
			// (1 - E_spec)(1 - T)(1 - M) weights fSubsurface gets below; coat
			// and fuzz are ignored on the underside per the spec's own
			// thin-walled approximation
			let ηEffThin: f32 = dispersedIor;
			let iorRatioThin: f32 = (
				(ηEffThin - 1.0f) / (ηEffThin + 1.0f)
			);
			let essThin: f32 = (
				bsdfMicrofacetDielectricAlbedo(
					abs(dot(nor, ωi)),
					bsdfOpenPbrCoatRoughenedRoughness(
						mat.specularRoughness, mat.coatRoughness, mat.coatWeight
					),
					iorRatioThin * iorRatioThin
				)
			);
			fTransmitted += (
				(
					(1.0f - essThin)
					* (1.0f - mat.transmissionWeight)
					* mat.subsurfaceWeight
					* (1.0f - mat.baseMetalness)
				)
				* bsdfOpenPbrThinWalledSubsurfaceEvaluateF(mat, nor, ωi, ωo, λ)
			);
		}
		return fTransmitted;
	}

	// -- relative ior
	// at the hero wavelength so reflection and transmission share n(λ)
	let ηEff: f32 = dispersedIor;
	let ηRel: f32 = isInsideMedium ? (1.0f / ηEff) : ηEff;
	let iorRatio: f32 = (ηEff - 1.0f) / (ηEff + 1.0f);
	let f0Eff: f32 = iorRatio * iorRatio;

	// -- glossy-diffuse base
	let fDiffuse: vkof::Spectrum = (
		bsdfOpenPbrGlossyDiffuseOrenNayerEvaluateF(mat, nor, ωi, ωo, λ)
	);

	// -- subsurface; thin-walled swaps the walk for the diffuse sheet, which
	// unlike the walk does have an instantaneous value on this side
	let fSubsurface: vkof::Spectrum = (
		mat.geometryThinWalled > 0.0f
		? bsdfOpenPbrThinWalledSubsurfaceEvaluateF(mat, nor, ωi, ωo, λ)
		: bsdfOpenPbrSubsurfaceEvaluateF(mat, nor, ωi, ωo)
	);

	// -- translucent specular
	let fTranslucentSpecular: vkof::Spectrum = vkof::Spectrum.zero();

	// -- conductor
	let fConductor: vkof::Spectrum = (
		bsdfOpenPbrFresnelMetallicEvaluateF(mat, frame, ωi, ωo, λ)
	);

	// -- specular reflection
	let fSpecular: vkof::Spectrum = (
		bsdfOpenPbrDielectricSpecularEvaluateF(mat, frame, ωi, ωo, ηRel, λ)
		* mat.specularColor
	);

	// -- compensated directional albedo
	// roughened by a rough coat to match fSpecular's actual lobe width
	let roughenedSpecularRoughness: f32 = (
		bsdfOpenPbrCoatRoughenedRoughness(
			mat.specularRoughness, mat.coatRoughness, mat.coatWeight
		)
	);
	let fSpecularDirectionalAlbedo: f32 = (
		bsdfMicrofacetDielectricAlbedo(
			dot(nor, ωi), roughenedSpecularRoughness, f0Eff
		)
	);

	// -- combination of lobe components
	let fTDielectricBase: vkof::Spectrum = (
		lerp(
			lerp(fDiffuse, fSubsurface, mat.subsurfaceWeight),
			fTranslucentSpecular,
			mat.transmissionWeight
		)
	);
	let fDielectricBase: vkof::Spectrum = (
		fSpecular + ((1.0f - fSpecularDirectionalAlbedo) * fTDielectricBase)
	);
	let fBaseSubstrate: vkof::Spectrum = (
		lerp(fDielectricBase, fConductor, mat.baseMetalness)
	);

	// -- coat layer over the base substrate; openpbr figure 96
	let fCoatedBase: vkof::Spectrum = (
		bsdfOpenPbrCoatEvaluateF(
			mat, coatFrame, ωi, ωo, fBaseSubstrate, isInsideMedium, λ
		)
	);

	// -- fuzz layer over the coated base; openpbr figure 96
	let fSurface: vkof::Spectrum = (
		bsdfOpenPbrFuzzEvaluateF(
			zeltnerLtcParamHandle, mat, fCoatedBase, nor, ωi, ωo, λ
		)
	);
	return fSurface;
}

// -----------------------------------------------------------------------------
// -- lobe selection
// -----------------------------------------------------------------------------

// which lobe bsdfOpenPbrSampleWo picked
public enum BsdfOpenPbrLobe {
	Coat,
	Specular,
	Diffuse,
	Fuzz,
	Transmission,
	SubsurfaceEntry,
};

// lobe selection probabilities
public struct BsdfOpenPbrLobeSelection {
	public f32 probabilityCoat;
	public f32 probabilitySpecular;
	public f32 probabilityDiffuse;
	public f32 probabilityFuzz;
	public f32 probabilityTransmission;
	public f32 probabilitySubsurfaceEntry;
	public f32 probabilityTotal;
	public f32 fuzzA;
	public f32 fuzzB;
};

// selects a lobe to sample from, returning the selection probabilities and
// zeltner fuzz lobe parameters for the selected \omega_i
public func bsdfOpenPbrLobeSelection(
	zeltnerLtcParamHandle: u32,
	mat: MaterialBackend_OpenPbr.Params,
	dotNorWi: f32,
) -> BsdfOpenPbrLobeSelection {
	let μ: f32 = max(dotNorWi, 1e-5f);
	let ηEff: f32 = bsdfOpenPbrSpecularEffectiveIor(mat);
	let iorRatio: f32 = (ηEff - 1.0f) / (ηEff + 1.0f);
	let f0: f32 = iorRatio * iorRatio;

	var sel: BsdfOpenPbrLobeSelection;

	// coat interface albedo with the coat ior and roughness
	let coatIorRatio: f32 = (mat.coatIor - 1.0f) / (mat.coatIor + 1.0f);
	let coatF0: f32 = coatIorRatio * coatIorRatio;
	sel.probabilityCoat = (
		mat.coatWeight
		* bsdfMicrofacetDielectricAlbedo(μ, mat.coatRoughness, coatF0)
	);

	// metalness overrides the dielectric interface albedo
	let roughenedSpecularRoughness: f32 = (
		bsdfOpenPbrCoatRoughenedRoughness(
			mat.specularRoughness, mat.coatRoughness, mat.coatWeight
		)
	);
	sel.probabilitySpecular = (
		max(
			bsdfMicrofacetDielectricAlbedo(μ, roughenedSpecularRoughness, f0),
			mat.baseMetalness
		)
	);
	let baseBudget: f32 = 1.0f - mat.baseMetalness;
	let opaqueBudget: f32 = baseBudget * (1.0f - mat.transmissionWeight);
	sel.probabilityDiffuse = opaqueBudget * (1.0f - mat.subsurfaceWeight);
	sel.probabilitySubsurfaceEntry = opaqueBudget * mat.subsurfaceWeight;
	sel.probabilityTransmission = baseBudget * mat.transmissionWeight;
	// zeltner sheen; selection probability from the fitted directional R(\mu)
	let fuzzParams: f32v3 = (
		mat.fuzzWeight > 0.0f
			? bsdfZeltnerFuzzLookup(zeltnerLtcParamHandle, μ, mat.fuzzRoughness)
			: f32v3(0.0f)
	);
	sel.probabilityFuzz = mat.fuzzWeight * fuzzParams.z;
	sel.probabilityTotal = (
		sel.probabilityCoat + sel.probabilitySpecular
		+ sel.probabilityDiffuse + sel.probabilityFuzz
		+ sel.probabilityTransmission + sel.probabilitySubsurfaceEntry
	);
	sel.fuzzA = fuzzParams.x;
	sel.fuzzB = fuzzParams.y;

	return sel;
}

public func bsdfOpenPbrSubsurfaceSlotPdf(
	mat: MaterialBackend_OpenPbr.Params,
	dotNorWo: f32,
) -> f32 {
	return (
		mat.geometryThinWalled > 0.0f
		? bsdfOpenPbrThinWalledSubsurfacePdf(
			dotNorWo, mat.subsurfaceScatterAnisotropy
		)
		: bsdfOpenPbrSubsurfaceEntryPdf(dotNorWo)
	);
}

// mixture pdf over all lobes selection
public func bsdfOpenPbrEvaluatePdfSelected(
	sel: BsdfOpenPbrLobeSelection,
	mat: MaterialBackend_OpenPbr.Params,
	frame: ShadingFrame,
	coatFrame: ShadingFrame,
	ωi: f32v3,
	ωo: f32v3,
	isInsideMedium: bool,
	dispersedIor: f32,
) -> f32 {
	let nor: f32v3 = frame.nor;
	let dotNorWo: f32 = dot(nor, ωo);
	let roughenedSpecularRoughness: f32 = (
		bsdfOpenPbrCoatRoughenedRoughness(
			mat.specularRoughness, mat.coatRoughness, mat.coatWeight
		)
	);
	let specα: f32v2 = (
		bsdfOpenPbrRoughnessAlpha(
			roughenedSpecularRoughness, mat.specularRoughnessAnisotropy
		)
	);
	let coatα: f32v2 = (
		bsdfOpenPbrRoughnessAlpha(mat.coatRoughness, mat.coatRoughnessAnisotropy)
	);
	let specFrame: ShadingFrame = (
		shadingFrameRotate(frame, mat.specularRoughnessAnisotropyRotation)
	);

	if (dotNorWo <= 0.0f) {
		if (mat.geometryThinWalled > 0.0f) {
			return (
				(
					sel.probabilityTransmission
					* bsdfOpenPbrTransmissionPdf(
						mat, frame, ωi, ωo, isInsideMedium, dispersedIor
					)
					+ (
						sel.probabilitySubsurfaceEntry > 0.0f
						? sel.probabilitySubsurfaceEntry
							* bsdfOpenPbrSubsurfaceSlotPdf(mat, dotNorWo)
						: 0.0f
					)
				) / sel.probabilityTotal
			);
		}
		return (
			(
				(
					sel.probabilityTransmission > 0.0f
					? sel.probabilityTransmission
						* bsdfOpenPbrTransmissionPdf(
							mat, frame, ωi, ωo, isInsideMedium, dispersedIor
						)
					: 0.0f
				)
				+ (
					sel.probabilitySpecular > 0.0f
					? sel.probabilitySpecular
						* bsdfMicrofacetGgxBoundedReflectPdfAniso(
							specFrame.tanX, specFrame.tanY, specFrame.nor,
							ωi, ωo, specα
						)
					: 0.0f
				)
				+ (
					sel.probabilityCoat > 0.0f
					? sel.probabilityCoat
						* bsdfMicrofacetGgxBoundedReflectPdfAniso(
							coatFrame.tanX, coatFrame.tanY, coatFrame.nor,
							ωi, ωo, coatα
						)
					: 0.0f
				)
				+ (
					sel.probabilitySubsurfaceEntry > 0.0f
					? sel.probabilitySubsurfaceEntry
						* bsdfOpenPbrSubsurfaceSlotPdf(mat, dotNorWo)
					: 0.0f
				)
			) / sel.probabilityTotal
		);
	}

	// reflection pdfs (bounded vndf, eto & tokuyoshi 2023)
	let specPdf: f32 = (
		bsdfMicrofacetGgxBoundedReflectPdfAniso(
			specFrame.tanX, specFrame.tanY, specFrame.nor, ωi, ωo, specα
		)
	);
	let coatPdf: f32 = (
		bsdfMicrofacetGgxBoundedReflectPdfAniso(
			coatFrame.tanX, coatFrame.tanY, coatFrame.nor, ωi, ωo, coatα
		)
	);
	let diffPdf: f32 = bsdfOpenPbrGlossyDiffusePdf(dotNorWo);
	let fuzzPdf: f32 = bsdfOpenPbrFuzzPdf(nor, ωi, ωo, sel.fuzzA, sel.fuzzB);
	let transmissionTirPdf: f32 = (
		bsdfOpenPbrTransmissionTirReflectPdf(
			mat, frame, ωi, ωo, isInsideMedium, dispersedIor
		)
	);

	return (
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
					* bsdfOpenPbrSubsurfaceSlotPdf(mat, dotNorWo)
				: 0.0f
			)
		) / sel.probabilityTotal
	);
}

public func bsdfOpenPbrEvaluatePdf(
	zeltnerLtcParamHandle: u32,
	mat: MaterialBackend_OpenPbr.Params,
	frame: ShadingFrame,
	coatFrame: ShadingFrame,
	ωi: f32v3,
	ωo: f32v3,
	isInsideMedium: bool,
	dispersedIor: f32,
) -> f32 {
	let sel: BsdfOpenPbrLobeSelection = (
		bsdfOpenPbrLobeSelection(zeltnerLtcParamHandle, mat, dot(frame.nor, ωi))
	);
	return (
		bsdfOpenPbrEvaluatePdfSelected(
			sel, mat, frame, coatFrame, ωi, ωo, isInsideMedium, dispersedIor
		)
	);
}

// ξ.z picks the lobe, ξ.xy the direction within it
public func bsdfOpenPbrSampleWo(
	zeltnerLtcParamHandle: u32,
	frame: ShadingFrame,
	coatFrame: ShadingFrame,
	ωi: f32v3,
	mat: MaterialBackend_OpenPbr.Params,
	ξ: f32v3,
	out pdf: f32,
	out sampledTransmission: bool,
	out sampledSubsurfaceEntry: bool,
	out pickedLobe: BsdfOpenPbrLobe,
	isInsideMedium: bool,
	dispersedIor: f32,
) -> f32v3 {
	let nor: f32v3 = frame.nor;
	// walk through the layers of materials picking sample
	let sel: BsdfOpenPbrLobeSelection = (
		bsdfOpenPbrLobeSelection(zeltnerLtcParamHandle, mat, dot(nor, ωi))
	);
	let u: f32 = ξ.z;

	let cdfCoat: f32 = sel.probabilityCoat / sel.probabilityTotal;
	let cdfSpecular: f32 = (
		(sel.probabilityCoat + sel.probabilitySpecular)
		/ sel.probabilityTotal
	);
	let cdfFuzz: f32 = (
		(
			sel.probabilityCoat
			+ sel.probabilitySpecular
			+ sel.probabilityFuzz
		) / sel.probabilityTotal
	);
	let cdfTransmission: f32 = (
		(
			sel.probabilityCoat
			+ sel.probabilitySpecular
			+ sel.probabilityFuzz
			+ sel.probabilityTransmission
		) / sel.probabilityTotal
	);
	let cdfSubsurfaceEntry: f32 = (
		(
			sel.probabilityCoat
			+ sel.probabilitySpecular
			+ sel.probabilityFuzz
			+ sel.probabilityTransmission
			+ sel.probabilitySubsurfaceEntry
		) / sel.probabilityTotal
	);

	var ωo: f32v3;
	sampledTransmission = false;
	sampledSubsurfaceEntry = false;
	if (u < cdfCoat) {
		pickedLobe = BsdfOpenPbrLobe::Coat;
		// bounded vndf (eto & tokuyoshi 2023)
		// must match bsdfOpenPbrEvaluatePdfSelected
		let coatα: f32v2 = (
			bsdfOpenPbrRoughnessAlpha(
				mat.coatRoughness, mat.coatRoughnessAnisotropy
			)
		);
		ωo = (
			bsdfMicrofacetSampleGgxBoundedWoAniso(
				coatFrame.tanX, coatFrame.tanY, coatFrame.nor, ωi, coatα, ξ.xy
			)
		);
	} else if (u < cdfSpecular) {
		pickedLobe = BsdfOpenPbrLobe::Specular;
		// coat-roughened base roughness
		// must match bsdfOpenPbrEvaluatePdfSelected
		let roughenedSpecularRoughness: f32 = (
			bsdfOpenPbrCoatRoughenedRoughness(
				mat.specularRoughness, mat.coatRoughness, mat.coatWeight
			)
		);
		let specα: f32v2 = (
			bsdfOpenPbrRoughnessAlpha(
				roughenedSpecularRoughness, mat.specularRoughnessAnisotropy
			)
		);
		// gltf interop anisotropy rotation
		// must match bsdfOpenPbrEvaluatePdfSelected
		let specFrame: ShadingFrame = (
			shadingFrameRotate(
				frame, mat.specularRoughnessAnisotropyRotation
			)
		);
		ωo = (
			bsdfMicrofacetSampleGgxBoundedWoAniso(
				specFrame.tanX, specFrame.tanY, specFrame.nor,
				ωi, specα, ξ.xy
			)
		);
	} else if (u < cdfFuzz) {
		pickedLobe = BsdfOpenPbrLobe::Fuzz;
		ωo = bsdfOpenPbrFuzzSampleWo(nor, ωi, sel.fuzzA, sel.fuzzB, ξ.xy);
	} else if (u < cdfTransmission) {
		pickedLobe = BsdfOpenPbrLobe::Transmission;
		if (mat.geometryThinWalled > 0.0f) {
			// thin shell (khr_materials_transmission without
			// khr_materials_volume)
			ωo = -ωi;
			sampledTransmission = true;
		} else {
			// unbounded vndf for slopes and frame
			// must match bsdfOpenPbrTransmissionPdf
			let transmissionα: f32v2 = (
				bsdfOpenPbrRoughnessAlpha(
					mat.specularRoughness, mat.specularRoughnessAnisotropy
				)
			);
			let transmissionFrame: ShadingFrame = (
				shadingFrameRotate(
					frame, mat.specularRoughnessAnisotropyRotation
				)
			);
			let h: f32v3 = (
				bsdfMicrofacetSampleGgxVndfAniso(
					transmissionFrame.tanX, transmissionFrame.tanY,
					transmissionFrame.nor, ωi, transmissionα, ξ.xy
				)
			);
			let ηI: f32 = isInsideMedium ? dispersedIor : 1.0f;
			let ηT: f32 = isInsideMedium ? 1.0f : dispersedIor;
			let refracted: f32v3 = refract(-ωi, h, ηI / ηT);
			if (length(refracted) > 0.5f) {
				ωo = refracted;
				sampledTransmission = true;
			} else {
				// total internal reflection
				ωo = normalize(2.0f * dot(ωi, h) * h - ωi);
			}
		}
	} else if (u < cdfSubsurfaceEntry) {
		pickedLobe = BsdfOpenPbrLobe::SubsurfaceEntry;
		if (mat.geometryThinWalled > 0.0f) {
			// a thin sheet has no interior to walk through
			let uSide: f32 = (
				(u - cdfTransmission)
				/ max(cdfSubsurfaceEntry - cdfTransmission, 1e-7f)
			);
			ωo = (
				bsdfOpenPbrThinWalledSubsurfaceSampleWo(
					nor, mat.subsurfaceScatterAnisotropy, ξ.xy, uSide
				)
			);
		} else {
			ωo = bsdfOpenPbrSubsurfaceEntrySampleWo(nor, ξ.xy);
			sampledSubsurfaceEntry = true;
		}
	} else {
		pickedLobe = BsdfOpenPbrLobe::Diffuse;
		ωo = bsdfOpenPbrGlossyDiffuseSampleWo(nor, ξ.xy);
	}
	pdf = (
		bsdfOpenPbrEvaluatePdfSelected(
			sel, mat, frame, coatFrame, ωi, ωo, isInsideMedium, dispersedIor
		)
	);
	return ωo;
}

// -----------------------------------------------------------------------------
// -- bsdf interface
// -----------------------------------------------------------------------------

public struct Bsdf_OpenPbr : I_Bsdf {
	public ShadingFrame frame;
	public ShadingFrame coatFrame;
	public vkof::Wavelengths λ;
	public MaterialBackend_OpenPbr.Params params;
	// true when the path arrives from inside the medium behind the surface
	public bool isInsideMedium;
	public f32 dispersedIor;
	public u32 zeltnerLtcParamHandle;

	public __init(
		ShadingFrame frame,
		ShadingFrame coatFrame,
		vkof::Wavelengths λ,
		MaterialBackend_OpenPbr.Params params,
		bool isInsideMedium,
		f32 dispersedIor,
	) {
		this.frame = frame;
		this.coatFrame = coatFrame;
		this.λ = λ;
		this.params = params;
		this.isInsideMedium = isInsideMedium;
		this.dispersedIor = dispersedIor;
		this.zeltnerLtcParamHandle = vkof::global.zeltnerLtcParamHandle;
	}

	public func evaluate(ωi: f32v3, ωo: f32v3) -> vkof::Spectrum {
		return (
			bsdfOpenPbrEvaluateF(
				zeltnerLtcParamHandle, frame, coatFrame, -ωi, ωo, params,
				isInsideMedium, dispersedIor, λ
			)
			* abs(dot(frame.nor, ωo))
		);
	}

	public func pdf(ωi: f32v3, ωo: f32v3) -> f32 {
		return (
			bsdfOpenPbrEvaluatePdf(
				zeltnerLtcParamHandle, params, frame, coatFrame, -ωi, ωo,
				isInsideMedium, dispersedIor
			)
		);
	}

	public func sample(ωi: f32v3, ξ: f32v3) -> BsdfSample {
		var pdf: f32;
		var sampledTransmission: bool;
		var sampledSubsurfaceEntry: bool;
		var pickedLobe: BsdfOpenPbrLobe;
		let ωo: f32v3 = (
			bsdfOpenPbrSampleWo(
				zeltnerLtcParamHandle, frame, coatFrame, -ωi, params, ξ, pdf,
				sampledTransmission, sampledSubsurfaceEntry, pickedLobe,
				isInsideMedium, dispersedIor
			)
		);
		let dotNorWo: f32 = dot(frame.nor, ωo);

		var sample: BsdfSample;
		sample.ωo = ωo;
		sample.pdf = pdf;

		// -- evaluate the sample's kind
		sample.kind = InlineExpr {
			if (sampledSubsurfaceEntry) { return BsdfSampleKind::SubsurfaceEntry; }
			if (sampledTransmission) {
				return (
					params.geometryThinWalled > 0.0f
					? BsdfSampleKind::ThinTransmission
					: BsdfSampleKind::Transmission
				);
			}
			if (pickedLobe == BsdfOpenPbrLobe::SubsurfaceEntry && dotNorWo < 0.0f) {
				return BsdfSampleKind::ThinTransmission;
			}
			return BsdfSampleKind::Reflection;
		};

		// -- delta lobes
		// a thin wall and a matched interface transmit along exactly -ωi
		sample.isDeltaDirac = (
			(sampledTransmission && params.geometryThinWalled > 0.0f)
			|| (sampledTransmission && dispersedIor == 1.0f)
			|| (
				params.specularRoughness <= 0.0f
				&& (pickedLobe == BsdfOpenPbrLobe::Specular || sampledTransmission)
			)
		);

		// -- weight
		// the subsurface walk replaces f · |cosθ| / pdf
		// only the entry lobe's selection probability is divided
		if (sample.kind == BsdfSampleKind::SubsurfaceEntry) {
			let sel: BsdfOpenPbrLobeSelection = (
				bsdfOpenPbrLobeSelection(
					zeltnerLtcParamHandle, params, dot(frame.nor, -ωi)
				)
			);
			sample.weight = vkof::Spectrum.one() * sel.probabilityTotal;
			return sample;
		}
		// handle a reflection that lands below the surface
		if (
			pdf <= 0.0f
			|| (sample.kind == BsdfSampleKind::Reflection && dotNorWo <= 0.0f)
		) {
			sample.weight = vkof::Spectrum.zero();
			return sample;
		}
		sample.weight = evaluate(ωi, ωo) / pdf;
		return sample;
	}
}

} // namespace vkof
