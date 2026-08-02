import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-coat";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 thetaSteps;
	u32 phiSteps;
	u32 count;
};

[[vk::push_constant]] PC pc;

// constants normally provided by util-material-openpbr.glsl; replicated
// here per the convention in metallic_lobe_furnace_integrate.comp (the
// full include chain pulls in every lobe, so the lobe-only prelude is
// duplicated instead)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// golden reference for openPbrCoatIblEvaluate (util-material-openpbr-ibl.glsl):
// numerically integrates the real openPbrCoatEvaluateF over the wo
// hemisphere for a FIXED wi, against a uniform white baseSubstrate = 1
// (the split-sum's own N=V=R approximation is exact under a uniform
// environment -- see openPbrCoatIblEvaluate's own comment -- so this
// isolates whether the closed form's algebra is correct, not whether the
// mirror-direction sampling approximation holds).
//
// subsurface_weight and transmission_weight are pinned to 0 (a plain
// dielectric/metal base) to keep the sweep dimensionality tractable; the
// darkening term's dependence on those two only enters through
// albedo_base, which the eval/closed-form share verbatim either way.
//
// per sample, 11 floats: muI, coatWeight, coatRoughness, coatIor,
// coatDarkening, coatColorScalar, specularRoughness, specularWeight,
// specularIor, baseMetalness, baseColorScalar

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 11u;
	const f32 muI = inBuf[base+0u];

	OpenPbrMaterial mat;
	mat.coatWeight = inBuf[base+1u];
	mat.coatRoughness = inBuf[base+2u];
	mat.coatRoughnessAnisotropy = 0.0f;
	mat.coatIor = inBuf[base+3u];
	mat.coatDarkening = inBuf[base+4u];
	mat.coatColor = f32v3(inBuf[base+5u]);
	mat.specularRoughness = inBuf[base+6u];
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.specularWeight = inBuf[base+7u];
	mat.specularIor = inBuf[base+8u];
	mat.baseMetalness = inBuf[base+9u];
	mat.baseWeight = 1.0f;
	mat.baseColor = f32v3(inBuf[base+10u]);
	mat.subsurfaceWeight = 0.0f;
	mat.subsurfaceColor = f32v3(0.0f);
	mat.transmissionWeight = 0.0f;
	mat.transmissionColor = f32v3(0.0f);

	const MaterialTableHandles tables = MaterialTableHandles(0u, 0u);
	ShadingFrame coatFrame;
	coatFrame.tanX = f32v3(1.0f, 0.0f, 0.0f);
	coatFrame.tanY = f32v3(0.0f, 1.0f, 0.0f);
	coatFrame.nor = f32v3(0.0f, 0.0f, 1.0f);

	const f32v3 wi = f32v3(sqrt(max(0.0f, 1.0f - muI * muI)), 0.0f, muI);
	// a lambertian brdf value with total albedo 1, not the albedo itself --
	// openPbrCoatEvaluateF's baseSubstrate is a per-solid-angle brdf value
	// (like openPbrEvaluateF's fBaseSubstrate), and this integrator
	// hemisphere-integrates its return value, so feeding in a bare 1.0
	// here would inject a spurious extra factor of pi (integral of
	// cosThetaO over the hemisphere)
	const f32v3 baseSubstrate = f32v3(1.0f / PI);

	const f32 dTheta = (PI * 0.5f) / f32(pc.thetaSteps);
	const f32 dPhi = (2.0f * PI) / f32(pc.phiSteps);

	f32 sum = 0.0f;
	for (u32 ti = 0u; ti < pc.thetaSteps; ++ti) {
		const f32 theta = (f32(ti) + 0.5f) * dTheta;
		const f32 cosThetaO = cos(theta);
		const f32 sinThetaO = sin(theta);
		for (u32 pi = 0u; pi < pc.phiSteps; ++pi) {
			const f32 phi = (f32(pi) + 0.5f) * dPhi;
			const f32v3 wo = (
				f32v3(sinThetaO * cos(phi), sinThetaO * sin(phi), cosThetaO)
			);
			const f32v3 f = (
				openPbrCoatEvaluateF(
					tables, mat, coatFrame, wi, wo, baseSubstrate,
					/*isInsideMedium=*/ false
				)
			);
			sum += f.x * cosThetaO * sinThetaO * dTheta * dPhi;
		}
	}
	((f32 *)(pc.outVa))[i] = sum;
}
