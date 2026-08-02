import "util-regir";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
	u64 guardsVa;
	u64 cellsVa;
};

[[vk::push_constant]] PC pc;

// util-random reaches for the bluenoise texture set; regir never calls that
// path, but the declaration has to exist for the include to compile

// the open-addressed table: insert, the collision walk, and lookup. every
// thread owns a private 16-slot table, so the atomic never races and the two
// languages see the same probe order -- a shared table would be
// scheduling-dependent and could not be diffed at all
static const u32 skSlots = 16u;
// over-allocated per slot so this shader does not couple to
// sizeof(GpuRegirCell)
static const u64 skCellByteStride = 128ul;
// 20 keys into 16 slots, so the last few exhaust the probe budget
static const u32 skInsertCount = 20u;
// the inserted keys, then 4 that were never inserted
static const u32 skLookupCount = 24u;

// in:  gridBase(3), normalBucket
// out: per insert (slot, probes), then per lookup (slot, probes)
[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 thread = gl_GlobalInvocationID.x;
	if (thread >= pc.threadCount) { return; }

	f32 * inBuf = (f32 *)(pc.inVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 i = thread * 4u;
	const u32 o = thread * (skInsertCount + skLookupCount) * 2u;

	u32 * guards = (
		(u32 *)(pc.guardsVa + u64(thread) * skSlots * 4ul)
	);
	GpuRegirCell * cells = (
		(GpuRegirCell *)(pc.cellsVa + u64(thread) * skSlots * skCellByteStride)
	);

	const i32v3 gridBase = i32v3(
		i32(inBuf[i + 0u]), i32(inBuf[i + 1u]), i32(inBuf[i + 2u])
	);
	const u32 bucket = u32(inBuf[i + 3u]);

	for (u32 k = 0u; k < skInsertCount; ++k) {
		GpuRegirCellKey key;
		key.gridCoord = gridBase + i32v3(i32(k), i32(k & 1u), 0);
		key.normalBucket = (bucket + k) % skRegirNormalBucketCount;
		u32 probes = 0u;
		const u32 slot = utilRegirInsert(guards, cells, skSlots, key, probes);
		outBuf[o + k * 2u + 0u] = (
			slot == skRegirSlotInvalid ? -1.0f : f32(slot)
		);
		outBuf[o + k * 2u + 1u] = f32(probes);
	}

	const u32 lookupBase = o + skInsertCount * 2u;
	for (u32 k = 0u; k < skLookupCount; ++k) {
		GpuRegirCellKey key;
		key.gridCoord = gridBase + i32v3(i32(k), i32(k & 1u), 0);
		key.normalBucket = (bucket + k) % skRegirNormalBucketCount;
		u32 probes = 0u;
		const u32 slot = utilRegirLookup(guards, cells, skSlots, key, probes);
		outBuf[lookupBase + k * 2u + 0u] = (
			slot == skRegirSlotInvalid ? -1.0f : f32(slot)
		);
		outBuf[lookupBase + k * 2u + 1u] = f32(probes);
	}
}
