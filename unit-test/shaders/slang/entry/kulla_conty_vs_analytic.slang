import "util-energy-compensation";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 tableHandle;
	u32 count;
};

[[vk::push_constant]] PC pc;

// per sample: f32v2(mu, roughness). output: f32v2(table, analytic)
// analytic uses f0=f90=1 (perfect fresnel single-scatter albedo), which is
// the same physical quantity the kulla-conty table stores

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	const f32v2 s = ((f32v2 *)(pc.inVa))[i];
	const f32 table = (
		utilMicrofacetDielectricEnergyCompensationKullaConty(
			pc.tableHandle, s.x, s.y
		)
	);
	const f32 analytic = utilMicrofacetGgxDirectionalAlbedo(s.x, s.y, 1.0f, 1.0f);
	((f32v2 *)(pc.outVa))[i] = f32v2(table, analytic);
}
