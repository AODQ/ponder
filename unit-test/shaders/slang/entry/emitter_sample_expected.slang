#include "../util-emitter-test.slang"
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
#include "../util-sampling-bins.slang"
import "util-material-openpbr-shared";


// expected side of the emitter sampler's chi-square: one thread per bin,
// numerically integrates the emitter strategy's solid-angle density over the
// bin. the density is found by brute-force intersecting every emitter
// triangle, which is only tractable because these are synthetic test scenes
// -- and is deliberately a different code path from the sampler's own
// triangle choice, so the two can actually disagree.

static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 bin = gl_GlobalInvocationID.x;
	if (bin >= BSDF_VERIFY_BIN_COUNT) {
		return;
	}
	const f32v3 nor = f32v3(0.0f, 1.0f, 0.0f);

	const u32 thetaBin = bin / BSDF_VERIFY_PHI_BINS;
	const u32 phiBin = bin % BSDF_VERIFY_PHI_BINS;
	const f32 theta0 = PI * f32(thetaBin) / f32(BSDF_VERIFY_THETA_BINS);
	const f32 theta1 = PI * f32(thetaBin + 1u) / f32(BSDF_VERIFY_THETA_BINS);
	const f32 phi0 = TAU * f32(phiBin) / f32(BSDF_VERIFY_PHI_BINS);
	const f32 phi1 = TAU * f32(phiBin + 1u) / f32(BSDF_VERIFY_PHI_BINS);

	const u32 subGrid = max(pc.subGrid, 1u);
	f32 integral = 0.0f;
	for (u32 j = 0u; j < subGrid; ++j) {
		const f32 theta = lerp(theta0, theta1, (f32(j) + 0.5f) / f32(subGrid));
		const f32 sinTheta = sin(theta);
		for (u32 k = 0u; k < subGrid; ++k) {
			const f32 phi = lerp(phi0, phi1, (f32(k) + 0.5f) / f32(subGrid));
			const f32v3 dir = bsdfVerifyDirectionFromAngles(nor, theta, phi);
			integral += emitterTestDirectionPdf(pc.from, dir) * sinTheta;
		}
	}
	integral *= (theta1 - theta0) * (phi1 - phi0) / f32(subGrid * subGrid);

	((f32 *)(pc.outVa))[bin] = integral * f32(pc.sampleCount);
}
