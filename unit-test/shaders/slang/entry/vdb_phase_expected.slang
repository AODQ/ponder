import "util-vdb";
#include "../util-sampling-bins.slang"

struct PC {
	u32v4 rootPad[8];
	u64 expectedVa;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 wi;
	f32 anisotropyHG;
	f32 anisotropyDraine;
	f32 draineAlpha;
	f32 draineWeight;
};

[[vk::push_constant]] PC pc;

// expected side of the fog phase function chi-square: one thread per bin,
// numerically integrates utilVdbPhaseFog (the real production mixture eval)
// over the bin's solid angle around pc.wi. same quadrature convention as
// mixture_expected.comp.

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 bin = gl_GlobalInvocationID.x;
	if (bin >= BSDF_VERIFY_BIN_COUNT) {
		return;
	}

	UtilVdbPhaseParams params;
	params.anisotropyHG = pc.anisotropyHG;
	params.anisotropyDraine = pc.anisotropyDraine;
	params.draineAlpha = pc.draineAlpha;
	params.draineWeight = pc.draineWeight;

	const u32 thetaBin = bin / BSDF_VERIFY_PHI_BINS;
	const u32 phiBin = bin % BSDF_VERIFY_PHI_BINS;
	const f32 theta0 = PI * f32(thetaBin) / f32(BSDF_VERIFY_THETA_BINS);
	const f32 theta1 = PI * f32(thetaBin + 1u) / f32(BSDF_VERIFY_THETA_BINS);
	const f32 phi0 = TAU * f32(phiBin) / f32(BSDF_VERIFY_PHI_BINS);
	const f32 phi1 = TAU * f32(phiBin + 1u) / f32(BSDF_VERIFY_PHI_BINS);

	// same 32x32 midpoint quadrature as mixture_expected.comp
	const u32 subGrid = 32u;
	f32 integral = 0.0f;
	for (u32 j = 0u; j < subGrid; ++j) {
		const f32 theta = lerp(theta0, theta1, (f32(j) + 0.5f) / f32(subGrid));
		const f32 sinTheta = sin(theta);
		for (u32 k = 0u; k < subGrid; ++k) {
			const f32 phi = lerp(phi0, phi1, (f32(k) + 0.5f) / f32(subGrid));
			const f32v3 dir = bsdfVerifyDirectionFromAngles(pc.wi, theta, phi);
			const f32 phaseVal = utilVdbPhaseFog(dot(pc.wi, dir), params);
			integral += phaseVal * sinTheta;
		}
	}
	integral *= (
		(theta1 - theta0) * (phi1 - phi0)
		/ f32(subGrid * subGrid)
	);

	const f32 scaled = integral * f32(pc.sampleCount);

	if (bin == 0u) {
		printf(
			"vdb_phase_expected bin0 write integral=%f scaled=%f",
			integral, scaled
		);
	}

	((f32 *)(pc.expectedVa))[bin] = scaled;
}
