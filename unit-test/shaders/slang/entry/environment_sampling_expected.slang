import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
#include "../util-sampling-bins.slang"
import "util-environment-map";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 expectedVa;
	EnvironmentMapHandles handles;
	u32 sampleCount;
	u32 seedSalt;
	f32 rotation;
};

[[vk::push_constant]] PC pc;

// expected side of the environment map's sampler-vs-pdf chi-square check
// (same technique as sampling_expected.comp, retargeted): one thread per
// bin, numerically integrates environmentMapPdf over the bin's solid angle
// and scales by the sample count. full-sphere distribution with no support
// boundary anywhere (unlike the bsdf lobes' cap-cut/tir curves), so a
// coarser subgrid than sampling_expected.comp's 32x32 is plenty.

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// seedSalt is unused here; declared so the block layout stays byte-identical
// to environment_sampling_histogram.comp's and both dispatches can share one
// C++ push struct (EnvSamplingPush in test-environment-sampling.cpp)

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

	// this bin's (theta, phi) rectangle
	const u32 thetaBin = bin / BSDF_VERIFY_PHI_BINS;
	const u32 phiBin = bin % BSDF_VERIFY_PHI_BINS;
	const f32 theta0 = PI * f32(thetaBin) / f32(BSDF_VERIFY_THETA_BINS);
	const f32 theta1 = PI * f32(thetaBin + 1u) / f32(BSDF_VERIFY_THETA_BINS);
	const f32 phi0 = TAU * f32(phiBin) / f32(BSDF_VERIFY_PHI_BINS);
	const f32 phi1 = TAU * f32(phiBin + 1u) / f32(BSDF_VERIFY_PHI_BINS);

	// midpoint over the subgrid; pdf(omega) * sin(theta), d_omega =
	// sin(theta) d_theta d_phi
	const u32 subGrid = 8u;
	f32 integral = 0.0f;
	for (u32 j = 0u; j < subGrid; ++j) {
		const f32 theta = lerp(theta0, theta1, (f32(j) + 0.5f) / f32(subGrid));
		const f32 sinTheta = sin(theta);
		for (u32 k = 0u; k < subGrid; ++k) {
			const f32 phi = lerp(phi0, phi1, (f32(k) + 0.5f) / f32(subGrid));
			// same frame + phi convention as the histogram
			const f32v3 wo = (
				bsdfVerifyDirectionFromAngles(nor, theta, phi)
			);
			integral += environmentMapPdf(pc.handles, wo, pc.rotation) * sinTheta;
		}
	}
	integral *= (
		(theta1 - theta0) * (phi1 - phi0) / f32(subGrid * subGrid)
	);

	((f32 *)(pc.expectedVa))[bin] = integral * f32(pc.sampleCount);
}
