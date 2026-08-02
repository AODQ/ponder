import "util-shading-frame";
import "util-fresnel";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-dielectric-specular";
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

// furnace test: numerically integrates the full lobe (fresnel * d * v *
// energy compensation) over the wo hemisphere under constant unit
// illumination -- reflected energy should not exceed 1. isotropic
// (anisotropy=0), no coat (coatWeight=0), scoped to just this lobe.
//
// per sample: f32v3(mu, roughness, etaRel)

static const f32 kPi = 3.14159265358979323846f;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	const f32v3 s = ((f32v3 *)(pc.inVa))[i];
	const f32 mu = s.x;
	const f32 roughness = s.y;
	const f32 etaRel = s.z;

	const f32v3 wi = f32v3(sqrt(max(0.0f, 1.0f - mu * mu)), 0.0f, mu);

	ShadingFrame frame;
	frame.tanX = f32v3(1.0f, 0.0f, 0.0f);
	frame.tanY = f32v3(0.0f, 1.0f, 0.0f);
	frame.nor = f32v3(0.0f, 0.0f, 1.0f);

	const f32 dTheta = (kPi * 0.5f) / f32(pc.thetaSteps);
	const f32 dPhi = (2.0f * kPi) / f32(pc.phiSteps);

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
			const f32v3 f = unittest_openPbrDielectricSpecularEvaluateF(
				roughness, 0.0f, 0.0f, 0.0f, 0.0f, frame, wi, wo, etaRel
			);
			sum += f.x * cosThetaO * sinThetaO * dTheta * dPhi;
		}
	}
	((f32 *)(pc.outVa))[i] = sum;
}
