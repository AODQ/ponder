import "util-global-pc";
import "util-regir-pc";

struct PC {
	GpuGlobalPc global;
};

[[vk::push_constant]] PC pc;

// regir pass A: zero the guard words.
//
// a guard of 0 means "empty slot", so zeroing the guards is enough to retire
// the whole table -- a stale cell payload underneath is unreachable. vkof has
// no buffer-fill entry point, hence a compute pass rather than vkCmdFillBuffer.
//
// (review note) this runs every frame for now, which pins the refresh interval
// at K=1: every cell is rebuilt from scratch each frame. that is the
// correct-but-expensive end of plan-many-lights.md section 6.1 and it is the
// right place to start, because it makes the "is the grid stale" question not
// exist while the estimator is being proven. the cheaper round-robin policy
// needs an eviction rule (that is what GpuRegirCell.lastFillIndex is for) and
// should wait until the cost is actually measured.

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const GpuGlobalExtended unf = (
		((GpuGlobalExtended *)(pc.global.extended))[0]
	);
	const u32 slot = gl_GlobalInvocationID.x;
	if (unf.regirEnabled == 0u || slot >= unf.regirCellCount) {
		return;
	}
	((u32 *)(unf.regirGuards))[slot] = 0u;
}
