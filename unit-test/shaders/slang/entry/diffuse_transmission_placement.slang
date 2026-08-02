import "util-material-openpbr";

struct PC {
	u32v4 rootPad[8];
	u64 outVa;
	u32 kullaContyEnergyHandle;
	u32 zeltnerLtcParamHandle;
	u32 mode;
	// outgoing directions (sweep) or theta bands (energy)
	u32 sampleCount;
	// incidence measured from the normal, radians
	f32 wiTheta;
	f32v3 baseColor;
	f32v3 subsurfaceColor;
	f32 subsurfaceWeight;
	f32 subsurfaceScatterAnisotropy;
	f32 geometryThinWalled;
	f32 specularIor;
	f32 specularRoughness;
	f32 baseDiffuseRoughness;
	f32 coatWeight;
	f32 coatRoughness;
	f32 fuzzWeight;
	f32 fuzzRoughness;
	// quadrature resolution for the energy mode; exposed so the caller can
	// prove a reported albedo is converged rather than an integration artefact
	u32 subTheta;
	u32 subPhi;
};

[[vk::push_constant]] PC pc;

// where the thin-walled subsurface sheet ends up once it is *assembled* into
// the full material, which is a different question from whether the sheet
// function itself is right.
//
// test-thin-walled-subsurface.cpp already verifies the lobe in isolation, and
// test-diffuse-transmission.cpp verifies the loader mapping. Neither runs
// openPbrEvaluateF, so nothing checks the step in between: subsurfaceWeight
// blending the sheet against the base diffuse lobe inside
// openPbrEvaluateF/fTDielectricBase, with the specular interface, coat and
// fuzz layered over it. The gltf split
//
//   reflected   = (1 - factor) * baseColor
//   transmitted = factor * diffuseTransmissionColor
//
// is a statement about that assembled material, not about the sheet.
//
// mode 0 sweeps the outgoing direction through both hemispheres in the wi
// plane; mode 1 integrates f |cos| over the whole sphere so the caller can
// compare directional albedo against what the material was authored to emit.

#define DT_PLACEMENT_MODE_SWEEP 0u
#define DT_PLACEMENT_MODE_ENERGY 1u

// every field written explicitly; unwritten members are uninitialized gpu
// memory, not zero
func fnMaterial(

) -> OpenPbrMaterial {
	OpenPbrMaterial mat;
	mat.baseWeight = 1.0f;
	mat.baseColor = pc.baseColor;
	mat.baseMetalness = 0.0f;
	mat.baseDiffuseRoughness = pc.baseDiffuseRoughness;
	mat.specularWeight = 1.0f;
	mat.specularColor = f32v3(1.0f);
	mat.specularRoughness = pc.specularRoughness;
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.specularIor = pc.specularIor;
	mat.transmissionWeight = 0.0f;
	mat.transmissionColor = f32v3(1.0f);
	mat.transmissionDepth = 0.0f;
	mat.transmissionScatter = f32v3(0.0f);
	mat.transmissionScatterAnisotropy = 0.0f;
	mat.transmissionDispersionScale = 0.0f;
	mat.transmissionDispersionAbbeNumber = 20.0f;
	mat.subsurfaceWeight = pc.subsurfaceWeight;
	mat.subsurfaceColor = pc.subsurfaceColor;
	mat.subsurfaceRadius = 1.0f;
	mat.subsurfaceRadiusScale = f32v3(1.0f);
	mat.subsurfaceScatterAnisotropy = pc.subsurfaceScatterAnisotropy;
	mat.coatWeight = pc.coatWeight;
	mat.coatColor = f32v3(1.0f);
	mat.coatRoughness = pc.coatRoughness;
	mat.coatRoughnessAnisotropy = 0.0f;
	mat.coatIor = 1.5f;
	mat.coatDarkening = 1.0f;
	mat.fuzzWeight = pc.fuzzWeight;
	mat.fuzzColor = f32v3(1.0f);
	mat.fuzzRoughness = pc.fuzzRoughness;
	mat.emissionLuminance = 0.0f;
	mat.emissionColor = f32v3(0.0f);
	mat.thinFilmWeight = 0.0f;
	mat.thinFilmThickness = 0.0f;
	mat.thinFilmIor = 1.5f;
	mat.geometryOpacity = 1.0f;
	mat.geometryThinWalled = pc.geometryThinWalled;
	mat.geometryNormalTexture = f32v3(0.0f, 0.0f, 1.0f);
	mat.geometryTangentTexture = f32v4(1.0f, 0.0f, 0.0f, 1.0f);
	mat.geometryCoatNormalTexture = f32v3(0.0f, 0.0f, 1.0f);
	mat.geometryCoatTangentTexture = f32v4(1.0f, 0.0f, 0.0f, 1.0f);
	mat.alphaCutoff = 0.5f;
	mat.alphaMode = 0u;
	return mat;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 tid = gl_GlobalInvocationID.x;
	if (tid >= pc.sampleCount) {
		return;
	}

	ShadingFrame frame;
	frame.tanX = f32v3(1.0f, 0.0f, 0.0f);
	frame.tanY = f32v3(0.0f, 1.0f, 0.0f);
	frame.nor = f32v3(0.0f, 0.0f, 1.0f);
	const ShadingFrame coatFrame = frame;

	MaterialTableHandles tables;
	tables.kullaContyEnergy = pc.kullaContyEnergyHandle;
	tables.zeltnerLtcParam = pc.zeltnerLtcParamHandle;

	const OpenPbrMaterial mat = fnMaterial();
	const f32v3 wi = f32v3(sin(pc.wiTheta), 0.0f, cos(pc.wiTheta));
	const f32 dispersedIor = openPbrTransmissionIor(mat);

	if (pc.mode == DT_PLACEMENT_MODE_SWEEP) {
		// theta over the full 0..pi range, so the same buffer carries the
		// reflected and the transmitted half of the lobe
		const f32 theta = PI * (f32(tid) + 0.5f) / f32(pc.sampleCount);
		const f32v3 wo = f32v3(sin(theta), 0.0f, cos(theta));
		const f32v3 f = (
			openPbrEvaluateF(
				tables, frame, coatFrame, wi, wo, mat,
				/*isInsideMedium=*/false, dispersedIor
			)
		);
		((f32 *)(pc.outVa))[tid * 3u + 0u] = f.x;
		((f32 *)(pc.outVa))[tid * 3u + 1u] = f.y;
		((f32 *)(pc.outVa))[tid * 3u + 2u] = f.z;
		return;
	}

	// -- energy: this thread owns one theta band of the full sphere and
	// integrates f |cos| over it. the caller sums the bands into a
	// directional albedo, and separately sums only the reflected or only
	// the transmitted bands
	const f32 theta0 = PI * f32(tid) / f32(pc.sampleCount);
	const f32 theta1 = PI * f32(tid + 1u) / f32(pc.sampleCount);
	const u32 subTheta = pc.subTheta;
	const u32 subPhi = pc.subPhi;
	const f32 dTheta = (theta1 - theta0) / f32(subTheta);
	const f32 dPhi = TAU / f32(subPhi);
	f32v3 sum = f32v3(0.0f);
	for (u32 j = 0u; j < subTheta; ++j) {
		const f32 theta = theta0 + (f32(j) + 0.5f) * dTheta;
		const f32 sinTheta = sin(theta);
		const f32 cosTheta = cos(theta);
		for (u32 k = 0u; k < subPhi; ++k) {
			const f32 phi = (f32(k) + 0.5f) * dPhi;
			const f32v3 wo = (
				f32v3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta)
			);
			const f32v3 f = (
				openPbrEvaluateF(
					tables, frame, coatFrame, wi, wo, mat,
					/*isInsideMedium=*/false, dispersedIor
				)
			);
			sum += f * abs(cosTheta) * sinTheta * dTheta * dPhi;
		}
	}
	((f32 *)(pc.outVa))[tid * 3u + 0u] = sum.x;
	((f32 *)(pc.outVa))[tid * 3u + 1u] = sum.y;
	((f32 *)(pc.outVa))[tid * 3u + 2u] = sum.z;
}
