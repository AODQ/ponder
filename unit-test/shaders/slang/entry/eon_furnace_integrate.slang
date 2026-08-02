import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-diffuse";
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
// material include chain pulls in every lobe, so the diffuse-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// furnace test: numerically integrates the eon diffuse lobe over the wo
// hemisphere under constant unit illumination. the whole point of the
// energy-preserving oren-nayer construction is that this integral is
// exactly 1 for baseColor = 1 at every incident angle and roughness.
// baseWeight fixed to 1, gray baseColor supplied per sample.
//
// per sample: f32v3(muI, sigma, colorScalar)

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	const f32v3 s = ((f32v3 *)(pc.inVa))[i];
	const f32 muI = s.x;

	OpenPbrMaterial mat;
	mat.baseDiffuseRoughness = s.y;
	mat.baseWeight = 1.0f;
	mat.baseColor = f32v3(s.z);

	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);
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
			const f32v3 f = (
				openPbrGlossyDiffuseOrenNayerEvaluateF(mat, nor, wi, wo)
			);
			sum += f.x * cosThetaO * sinThetaO * dTheta * dPhi;
		}
	}
	((f32 *)(pc.outVa))[i] = sum;
}
