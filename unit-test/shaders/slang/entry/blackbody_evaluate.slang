import "util-blackbody";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// direct evaluation of utilBlackbodyRadiance, one temperature (kelvin) per
// sample

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	const f32 temperatureKelvin = ((f32 *)(pc.inVa))[i];
	((f32v3 *)(pc.outVa))[i] = utilBlackbodyRadiance(temperatureKelvin);
}
