import "util-vdb";
import "util-global-pc";

struct PC {
	u32v4 rootPad[8];
	u64 vdbVa;
	u64 pointsVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// gpu half of the vdb density crosscheck: reads utilVdbDensity at
// host-provided world positions so the host can compare against cpu-side
// nanovdb accessor reads of the byte-identical blob
// (test-vdb-sampling.cpp). exercises the full chain the viewer relies on:
// blob upload -> bda bind -> pnanovdb tree traversal -> world-to-index
// transform -> voxel read

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	UtilVdb vdb = utilVdbBind(pc.vdbVa);
	f32v3 * pointBuf = (f32v3 *)(pc.pointsVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[i] = utilVdbDensity(vdb, pointBuf[i]);
}
