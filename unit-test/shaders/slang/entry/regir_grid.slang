#include "../util-regir-test.slang"


// tier 9 of plan-many-lights.md: the regir hash grid, with no reservoirs in
// it yet. three modes over one dispatch --
//
//   key    -- derive (gridCoord, normalBucket, hashKey) for each query point
//             and report all three, so a cpu mirror can crosscheck the exact
//             integers rather than only the observable behaviour
//   insert -- claim a slot per key, report slot and probe count
//   lookup -- find a key's slot without claiming, report slot and probe count
//
// insert and lookup are separate dispatches on purpose: the renderer runs them
// in separate passes with a barrier between (plan section "concurrency"), and a
// test that interleaved them would be testing something the renderer never does.

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 index = gl_GlobalInvocationID.x;
	if (index >= pc.count) {
		return;
	}

	const RegirTestPoint point = ((RegirTestPoint *)(pc.inVa))[index];
	u32 * outBuf = (u32 *)(pc.outVa);

	const i32v3 gridCoord = utilRegirGridCoord(point.position, pc.cellSize);
	const u32 normalBucket = utilRegirNormalBucket(point.normal);

	GpuRegirCellKey key;
	key.gridCoord = gridCoord;
	key.normalBucket = normalBucket;

	if (pc.mode == REGIR_TEST_MODE_KEY) {
		const u32 base = index * REGIR_TEST_KEY_STRIDE;
		// bit-cast rather than convert: a negative coordinate must survive the
		// readback intact, and that is exactly the case being tested
		outBuf[base + 0u] = u32(gridCoord.x);
		outBuf[base + 1u] = u32(gridCoord.y);
		outBuf[base + 2u] = u32(gridCoord.z);
		outBuf[base + 3u] = normalBucket;
		outBuf[base + 4u] = utilRegirCellHashKey(key);
		return;
	}

	const u32 slotBase = index * REGIR_TEST_SLOT_STRIDE;
	u32 probes = 0u;
	u32 slot;
	if (pc.mode == REGIR_TEST_MODE_INSERT) {
		slot = utilRegirInsert(
			(u32 *)(pc.guardsVa),
			(GpuRegirCell *)(pc.cellsVa),
			pc.cellCount, key, probes
		);
	} else {
		slot = utilRegirLookup(
			(u32 *)(pc.guardsVa),
			(GpuRegirCell *)(pc.cellsVa),
			pc.cellCount, key, probes
		);
	}
	outBuf[slotBase + 0u] = slot;
	outBuf[slotBase + 1u] = probes;
}
