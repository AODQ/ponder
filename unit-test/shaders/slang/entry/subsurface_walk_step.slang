import "util-material-openpbr-tables";
import "util-material-openpbr-subsurface";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 outcomeVa;
	u64 distanceVa;
	u64 woVa;
	u64 weightVa;
	u32 sampleCount;
	u32 seedSalt;
	f32v3 sigmaAbsorption;
	f32v3 sigmaScattering;
	f32 distanceMax;
	f32v3 wi;
	f32 anisotropy;
	f32v3 throughput;
};

[[vk::push_constant]] PC pc;

// sampler side of openPbrSubsurfaceWalkStep
// (util-material-openpbr-subsurface.glsl): one thread per sample, all
// samples drawn from the same fixed (sigmaAbsorption, sigmaScattering,
// distanceMax, wi, anisotropy, throughput) point so the outcomes can be
// checked statistically against the closed-form exponential/HG/adaptive-
// mis predictions on the cpu side. ported 1:1 from blender cycles'
// volume_sample_channel / volume_sample_channel_pdf +
// subsurface_random_walk_pdf (see openPbrSubsurfaceWalkStep's header
// comment for the exact source files).

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 sampleIndex = gl_GlobalInvocationID.x;
	if (sampleIndex >= pc.sampleCount) {
		return;
	}

	const u32 h0 = fnPcgHash(sampleIndex + fnPcgHash(pc.seedSalt));
	const u32 h1 = fnPcgHash(h0);
	u64 state = (u64(h0) << 32) | u64(h1);

	f32 outDistance;
	f32v3 outWo;
	f32v3 outWeight;
	const u32 outcome = (
		openPbrSubsurfaceWalkStep(
			pc.sigmaAbsorption, pc.sigmaScattering, pc.distanceMax,
			pc.wi, pc.anisotropy, pc.throughput,
			// guiding disabled: this test covers the classical chromatic
			// weighting in isolation
			f32v3(0.0f),
			outDistance, outWo, outWeight, state
		)
	);

	((u32 *)(pc.outcomeVa))[sampleIndex] = outcome;
	((f32 *)(pc.distanceVa))[sampleIndex] = outDistance;
	((f32v3 *)(pc.woVa))[sampleIndex] = outWo;
	((f32v3 *)(pc.weightVa))[sampleIndex] = outWeight;
}
