import "util-material-openpbr-tables";
import "util-material-openpbr-subsurface";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 pointCount;
};

[[vk::push_constant]] PC pc;

// evaluate shader for openPbrSubsurfaceAlbedoToSigma
// (util-material-openpbr-subsurface.glsl): the diffusion-inversion that
// turns openpbr's user-facing (subsurfaceColor, subsurfaceRadius *
// subsurfaceRadiusScale, subsurfaceScatterAnisotropy) into the
// (sigmaAbsorption, sigmaScattering) pair a random walk integrates with.

// per point, 2 vec4s in: (albedo.rgb, anisotropy), (radius.rgb, unused).
// output 2 vec4s: (sigmaAbsorption.rgb, unused), (sigmaScattering.rgb, unused)

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.pointCount) {
		return;
	}

	const f32v4 in0 = ((f32v4 *)(pc.inVa))[i * 2u + 0u];
	const f32v4 in1 = ((f32v4 *)(pc.inVa))[i * 2u + 1u];
	const f32v3 albedo = in0.xyz;
	const f32 anisotropy = in0.w;
	const f32v3 radius = in1.xyz;

	f32v3 sigmaAbsorption;
	f32v3 sigmaScattering;
	openPbrSubsurfaceAlbedoToSigma(
		albedo, radius, anisotropy, sigmaAbsorption, sigmaScattering
	);

	((f32v4 *)(pc.outVa))[i * 2u + 0u] = f32v4(sigmaAbsorption, 0.0f);
	((f32v4 *)(pc.outVa))[i * 2u + 1u] = f32v4(sigmaScattering, 0.0f);
}
