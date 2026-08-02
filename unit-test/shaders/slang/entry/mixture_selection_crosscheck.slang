import "util-random";
import "util-material-openpbr";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 mineOutVa;
	u64 refOutVa;
	u32 zeltnerLtcParamHandle;
	u32 count;
};

[[vk::push_constant]] PC pc;

// independent cross-check of openPbrLobeSelection's *combination* arithmetic
// (the budget split / max / sum that turns five per-lobe weights into
// selection probabilities), not of the per-lobe weight formulas themselves.
// utilMicrofacetDielectricAlbedo (coat/specular dielectric albedo),
// utilZeltnerFuzzLookup (fuzz R channel), openPbrCoatRoughenedRoughness and
// openPbrEffectiveIor are called by both "mine" and "ref" below and are
// treated as already-independently-verified trusted inputs (dielectric-
// specular, fuzz and metallic milestones) -- there is no materialx mixture-
// bsdf equivalent to re-derive them against here, and the brief for this
// milestone explicitly scopes re-verification to the combination, not the
// primitives. the ref side is re-typed from the openpbr spec's stated
// layering split (opaque base = mix(diffuse, translucent, T); dielectric
// base under coat/specular/fuzz) rather than copied from
// openPbrLobeSelection's glsl, to catch a transcription slip (wrong
// variable, wrong operator, swapped budget term) in the original -- the
// formula itself is short enough that there isn't a materially different
// way to derive it, so independence here is weaker than the transmission/
// metallic crosschecks; documented rather than overstated.
func openPbrLobeSelectionRef(
	tables: MaterialTableHandles,
	mu: f32,
	coatWeight: f32,
	coatRoughness: f32,
	coatIor: f32,
	specularWeight: f32,
	specularIor: f32,
	specularRoughness: f32,
	baseMetalness: f32,
	transmissionWeight: f32,
	fuzzWeight: f32,
	fuzzRoughness: f32,
) -> OpenPbrLobeSelection {
	const f32 refCoatIorRatio = (coatIor - 1.0f) / (coatIor + 1.0f);
	const f32 refCoatF0 = refCoatIorRatio * refCoatIorRatio;
	const f32 refCoat = (
		coatWeight
		* utilMicrofacetDielectricAlbedo(mu, coatRoughness, refCoatF0)
	);

	// the base is immersed in the coat, so \eta_s is the coat-affected ratio
	// before specular_weight's f0 modulation (openpbr specular_ior_ratio);
	// re-derived from the scalars here rather than calling the mat-level
	// openPbrSpecularEffectiveIor, so this stays an independent combination
	const f32 refEtaEff = (
		openPbrEffectiveIor(
			openPbrCoatAffectedIor(specularIor, coatIor, coatWeight),
			specularWeight
		)
	);
	const f32 refIorRatio = (refEtaEff - 1.0f) / (refEtaEff + 1.0f);
	const f32 refF0 = refIorRatio * refIorRatio;
	const f32 refRoughenedSpecRoughness = (
		openPbrCoatRoughenedRoughness(specularRoughness, coatRoughness, coatWeight)
	);
	const f32 refSpecDielectric = (
		utilMicrofacetDielectricAlbedo(mu, refRoughenedSpecRoughness, refF0)
	);
	const f32 refSpecular = max(refSpecDielectric, baseMetalness);

	// the base budget remaining once metalness has taken its share, split
	// between the opaque (diffuse) and translucent (transmission) base by
	// transmission_weight -- openpbr spec section on base substrate mixing
	const f32 refBaseBudget = 1.0f - baseMetalness;
	const f32 refDiffuse = refBaseBudget * (1.0f - transmissionWeight);
	const f32 refTransmission = refBaseBudget * transmissionWeight;

	const f32v3 refFuzzParams = (
		utilZeltnerFuzzLookup(tables, mu, fuzzRoughness)
	);
	const f32 refFuzz = fuzzWeight * refFuzzParams.z;

	OpenPbrLobeSelection sel;
	sel.probabilityCoat = refCoat;
	sel.probabilitySpecular = refSpecular;
	sel.probabilityDiffuse = refDiffuse;
	sel.probabilityFuzz = refFuzz;
	sel.probabilityTransmission = refTransmission;
	sel.probabilityTotal = (
		refCoat + refSpecular + refDiffuse + refFuzz + refTransmission
	);
	sel.fuzzA = refFuzzParams.x;
	sel.fuzzB = refFuzzParams.y;
	return sel;
}

// per config, 11 floats in: mu, coatWeight, coatRoughness, coatIor,
// specularWeight, specularIor, specularRoughness, baseMetalness,
// transmissionWeight, fuzzWeight, fuzzRoughness.
// output, 6 floats each for mine/ref: probabilityCoat, probabilitySpecular,
// probabilityDiffuse, probabilityFuzz, probabilityTransmission,
// probabilityTotal

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }

	f32 * inBuf = (f32 *)(pc.inVa);
	const u32 base = i * 11u;
	const f32 mu = inBuf[base+0u];
	const f32 coatWeight = inBuf[base+1u];
	const f32 coatRoughness = inBuf[base+2u];
	const f32 coatIor = inBuf[base+3u];
	const f32 specularWeight = inBuf[base+4u];
	const f32 specularIor = inBuf[base+5u];
	const f32 specularRoughness = inBuf[base+6u];
	const f32 baseMetalness = inBuf[base+7u];
	const f32 transmissionWeight = inBuf[base+8u];
	const f32 fuzzWeight = inBuf[base+9u];
	const f32 fuzzRoughness = inBuf[base+10u];

	MaterialTableHandles tables;
	tables.kullaContyEnergy = 0u;
	tables.zeltnerLtcParam = pc.zeltnerLtcParamHandle;

	OpenPbrMaterial mat;
	mat.coatWeight = coatWeight;
	mat.coatRoughness = coatRoughness;
	mat.coatIor = coatIor;
	mat.specularWeight = specularWeight;
	mat.specularIor = specularIor;
	mat.specularRoughness = specularRoughness;
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.baseMetalness = baseMetalness;
	mat.transmissionWeight = transmissionWeight;
	// openPbrLobeSelection now reads this to split the diffuse budget; left
	// unset it's uninitialized gpu memory, not a guaranteed 0
	mat.subsurfaceWeight = 0.0f;
	mat.fuzzWeight = fuzzWeight;
	mat.fuzzRoughness = fuzzRoughness;

	const OpenPbrLobeSelection mine = openPbrLobeSelection(tables, mat, mu);
	const OpenPbrLobeSelection ref = (
		openPbrLobeSelectionRef(
			tables, mu,
			coatWeight, coatRoughness, coatIor,
			specularWeight, specularIor, specularRoughness,
			baseMetalness, transmissionWeight, fuzzWeight, fuzzRoughness
		)
	);

	f32 * mineOut = (f32 *)(pc.mineOutVa);
	const u32 outBase = i * 6u;
	mineOut[outBase+0u] = mine.probabilityCoat;
	mineOut[outBase+1u] = mine.probabilitySpecular;
	mineOut[outBase+2u] = mine.probabilityDiffuse;
	mineOut[outBase+3u] = mine.probabilityFuzz;
	mineOut[outBase+4u] = mine.probabilityTransmission;
	mineOut[outBase+5u] = mine.probabilityTotal;

	f32 * refOut = (f32 *)(pc.refOutVa);
	refOut[outBase+0u] = ref.probabilityCoat;
	refOut[outBase+1u] = ref.probabilitySpecular;
	refOut[outBase+2u] = ref.probabilityDiffuse;
	refOut[outBase+3u] = ref.probabilityFuzz;
	refOut[outBase+4u] = ref.probabilityTransmission;
	refOut[outBase+5u] = ref.probabilityTotal;
}
