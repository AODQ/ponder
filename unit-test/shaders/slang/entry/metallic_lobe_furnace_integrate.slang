import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-metallic";
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

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// furnace test: numerically integrates the conductor lobe over the wo
// hemisphere for a FIXED wi (incident direction), under constant unit
// illumination. the conductor's fresnel (F82, evaluated at dot(nor, wi))
// and its turquin compensation factor (also a function of dot(nor, wi)
// only) are both constant across this integral, so they factor out:
//   integral_wo [F82(muI) * D * V * comp(muI)] cos dwo
//     = F82(muI) * comp(muI) * E_actual(muI)
// with comp = 1 + fAvg (1-E_fit)/E_fit; to the extent E_fit tracks
// E_actual this reduces to F82(muI) * (E + fAvg (1-E)), which for a fully
// white metal (baseWeight=baseColor=1 -> f0=1 -> F82=1, fAvg=1) is exactly
// 1: the turquin compensation's entire purpose. note this furnace fixes
// wi, unlike the glossy-diffuse layer furnace which fixes wo -- each
// lobe's identity factors on the side its albedo terms are evaluated at.
//
// per sample, 8 floats: muI (fixed incident cosine), specularRoughness,
// specularRoughnessAnisotropy, baseWeight, baseColorScalar,
// specularWeight, specularColorScalar, unused

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 8u;
	const f32 muI = inBuf[base+0u];

	OpenPbrMaterial mat;
	mat.specularRoughness = inBuf[base+1u];
	mat.specularRoughnessAnisotropy = inBuf[base+2u];
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.baseWeight = inBuf[base+3u];
	mat.baseColor = f32v3(inBuf[base+4u]);
	mat.specularWeight = inBuf[base+5u];
	mat.specularColor = f32v3(inBuf[base+6u]);
	mat.coatWeight = 0.0f;
	mat.coatRoughness = 0.0f;
	mat.thinFilmWeight = 0.0f;

	ShadingFrame frame;
	frame.tanX = f32v3(1.0f, 0.0f, 0.0f);
	frame.tanY = f32v3(0.0f, 1.0f, 0.0f);
	frame.nor = f32v3(0.0f, 0.0f, 1.0f);

	const f32v3 wi = f32v3(sqrt(max(0.0f, 1.0f - muI * muI)), 0.0f, muI);

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
			const f32v3 f = openPbrFresnelMetallicEvaluateF(mat, frame, wi, wo);
			sum += f.x * cosThetaO * sinThetaO * dTheta * dPhi;
		}
	}
	((f32 *)(pc.outVa))[i] = sum;
}
