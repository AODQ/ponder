import "util-regir";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
};

[[vk::push_constant]] PC pc;

// util-random reaches for the bluenoise texture set; regir never calls that
// path, but the declaration has to exist for the include to compile

// the pure half of util-regir: key derivation, the probe walk's slot
// arithmetic, and the reservoir primitives. insert/lookup/select need real
// device buffers and are not covered here
static const u32 skCellCount = 1024u;

// in:  worldPosition(3), normal(3), cellSize, hasNormal, probe, weight,
//      target, xi
// out: gridCoord(3), oct(2), normalBucket, hashLow, hashHigh, keyEqualSelf,
//      keyEqualOther, slot, reservoir(6), sampleCount
[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 thread = gl_GlobalInvocationID.x;
	if (thread >= pc.threadCount) { return; }

	f32 * inBuf = (f32 *)(pc.inVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 i = thread * 12u;
	const u32 o = thread * 18u;

	const f32v3 worldPosition = f32v3(
		inBuf[i + 0u], inBuf[i + 1u], inBuf[i + 2u]
	);
	const f32v3 normal = normalize(
		f32v3(inBuf[i + 3u], inBuf[i + 4u], inBuf[i + 5u])
	);
	const f32 cellSize = inBuf[i + 6u];
	const bool hasNormal = inBuf[i + 7u] > 0.5f;
	const u32 probe = u32(inBuf[i + 8u]);
	const f32 weightBase = inBuf[i + 9u];
	const f32 targetBase = inBuf[i + 10u];
	const f32 xiBase = inBuf[i + 11u];

	const i32v3 gridCoord = utilRegirGridCoord(worldPosition, cellSize);
	outBuf[o + 0u] = f32(gridCoord.x);
	outBuf[o + 1u] = f32(gridCoord.y);
	outBuf[o + 2u] = f32(gridCoord.z);

	const f32v2 oct = utilRegirOctahedronNormalEncode(normal);
	outBuf[o + 3u] = oct.x;
	outBuf[o + 4u] = oct.y;
	outBuf[o + 5u] = f32(utilRegirNormalBucket(normal));

	const GpuRegirCellKey key = utilRegirKeyFor(
		worldPosition, normal, hasNormal, cellSize
	);
	// the hash is a full 32 bits, which f32 cannot hold exactly; each half can
	const u32 hashKey = utilRegirCellHashKey(key);
	outBuf[o + 6u] = f32(hashKey & 0xffffu);
	outBuf[o + 7u] = f32(hashKey >> 16u);

	// keyOther differs in exactly one grid axis on odd probes and in the
	// bucket alone on every second pair, so a componentwise compare that is
	// not folded to a single bool disagrees here
	GpuRegirCellKey keyOther = key;
	keyOther.gridCoord.y += i32(probe & 1u);
	keyOther.normalBucket += (probe >> 1u) & 1u;
	outBuf[o + 8u] = utilRegirKeyEqual(key, key) ? 1.0f : 0.0f;
	outBuf[o + 9u] = utilRegirKeyEqual(key, keyOther) ? 1.0f : 0.0f;

	outBuf[o + 10u] = f32(utilRegirSlotAt(hashKey, probe, skCellCount));

	// four candidates: the first two are non-positive so the count-but-skip
	// path is exercised on every lane
	RegirReservoirState res = utilRegirReservoirInit();
	for (u32 c = 0u; c < 4u; ++c) {
		utilRegirReservoirUpdate(
			res,
			c,
			c * 7u,
			f32v2(f32(c) * 0.1f, f32(c) * 0.2f),
			weightBase * (f32(c) - 1.0f),
			targetBase * f32(c + 1u),
			xiBase * 0.25f * f32(c + 1u)
		);
	}
	const GpuRegirReservoir stored = utilRegirReservoirFinalize(res);
	outBuf[o + 11u] = f32(stored.lightIndex);
	outBuf[o + 12u] = f32(stored.primitiveIndex);
	outBuf[o + 13u] = stored.barycentric.x;
	outBuf[o + 14u] = stored.barycentric.y;
	outBuf[o + 15u] = stored.targetValue;
	outBuf[o + 16u] = stored.unbiasedWeight;
	outBuf[o + 17u] = f32(res.sampleCount);
}
