import "util-vdb";

struct PC {
	u32v4 rootPad[8];
	u64 outVa;
	u32 xiCount;
	u32 gCount;
	u32 alphaCount;
	f32 gMin;
	f32 gMax;
	f32 alphaMin;
	f32 alphaMax;
};

[[vk::push_constant]] PC pc;

// numerical-robustness sweep for utilVdbPhaseSampleDraineCos's cubic-root
// inversion (jendersie & d'eon 2023): the closed-form solve involves large
// cancelling terms (432*T1a^3 + T2 + ...) that can lose precision or go
// non-finite for some (xi, g, alpha) combinations even though the analytic
// formula is exact. one thread per grid cell over xi x g x alpha; alpha
// range covers up to the fitted fog values (utilVdbPhaseFogDefaultParams
// reaches alpha ~27 around a 20 micron droplet diameter).

[shader("compute")]
[numthreads(8, 8, 4)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 xiIndex = gl_GlobalInvocationID.x;
	const u32 gIndex = gl_GlobalInvocationID.y;
	const u32 alphaIndex = gl_GlobalInvocationID.z;
	if (xiIndex >= pc.xiCount || gIndex >= pc.gCount || alphaIndex >= pc.alphaCount) {
		return;
	}

	// avoid exactly 0/1 (degenerate endpoints) and exactly g=0 (the sampler's
	// own division by g there is a separate, already-guarded isotropic path)
	const f32 xi = (f32(xiIndex) + 0.5f) / f32(pc.xiCount);
	const f32 g = lerp(pc.gMin, pc.gMax, (f32(gIndex) + 0.5f) / f32(pc.gCount));
	const f32 alpha = (
		lerp(pc.alphaMin, pc.alphaMax, (f32(alphaIndex) + 0.5f) / f32(pc.alphaCount))
	);

	const f32 cosTheta = utilVdbPhaseSampleDraineCos(xi, g, alpha);

	const u32 index = (
		(alphaIndex * pc.gCount + gIndex) * pc.xiCount + xiIndex
	);
	((f32 *)(pc.outVa))[index] = cosTheta;
}
