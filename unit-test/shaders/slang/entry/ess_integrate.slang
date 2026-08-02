#include "../util-microfacet-prelude.slang"

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 thetaSteps;
	u32 phiSteps;
	u32 count;
};

[[vk::push_constant]] PC pc;

// golden reference: numerically integrates the true ggx single-scatter
// directional albedo (perfect fresnel, F=1) from first principles, using
// the already-validated utilMicrofacetGgxDistribution and
// utilMicrofacetSmithGgxVisibility -- not copied from cull, materialx, or
// the kulla-conty table; independently derived to check both of those.
//
//   E_ss(mu, alpha) = integral_hemisphere D(h) * V(wi,wo) * cosThetaO dwo
//
// per sample: f32v2(mu, alpha)

static const f32 kPi = 3.14159265358979323846f;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	const f32v2 s = ((f32v2 *)(pc.inVa))[i];
	const f32 mu = s.x;
	const f32 alpha = s.y;

	const f32v3 wi = f32v3(sqrt(max(0.0f, 1.0f - mu * mu)), 0.0f, mu);

	const f32 dTheta = (kPi * 0.5f) / f32(pc.thetaSteps);
	const f32 dPhi = (2.0f * kPi) / f32(pc.phiSteps);

	f32 sum = 0.0f;
	for (u32 ti = 0u; ti < pc.thetaSteps; ++ti) {
		const f32 theta = (f32(ti) + 0.5f) * dTheta;
		const f32 cosThetaO = cos(theta);
		const f32 sinThetaO = sin(theta);
		for (u32 pi = 0u; pi < pc.phiSteps; ++pi) {
			const f32 phi = (f32(pi) + 0.5f) * dPhi;
			const f32v3 wo = f32v3(sinThetaO * cos(phi), sinThetaO * sin(phi), cosThetaO);
			const f32v3 h = normalize(wi + wo);
			const f32 D = utilMicrofacetGgxDistribution(h.z, alpha);
			const f32 V = utilMicrofacetSmithGgxVisibility(mu, cosThetaO, alpha);
			sum += D * V * cosThetaO * sinThetaO * dTheta * dPhi;
		}
	}
	((f32 *)(pc.outVa))[i] = sum;
}
