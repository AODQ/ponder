implementing types;

namespace vkof {

// 16 octahedral direction buckets, one reserved for medium vertices
public static const u32 skRegirNormalResolution = 4u;
public static const u32 skRegirNormalBucketCount = (
	skRegirNormalResolution * skRegirNormalResolution
);
// medium scatter has no normal, so reserve just one bucket for it
public static const u32 skRegirNormalBucketNone = skRegirNormalBucketCount;

public static const u32 skRegirMaxProbesPerCell = 32u;
public static const u32 skRegirSlotInvalid = 0xffffffffu;
public static const u32 skRegirSlotEvicted = 0xfffffffeu;

// reservoir index: slot*skRegirReservoirCount + index
public static const u32 skRegirReservoirCount = 64u;
public static const u32 skRegirCandidateCount = 32u;
// reservoirs resampled at the shading point
public static const u32 skRegirShadeCandidateCount = 8u;

public struct RegirCellKey {
	public i32v3 gridCoord;
	// octahedral bucket
	public u32 normalBucket;
};

// a reservoir of a single light sample
public struct RegirReservoir {
	public u32 lightIndex;
	public u32 primitiveIndex;
	// parameters on surface, e.g. barycentrics for triangle
	public f32v2 parameters;
	public f32 weight;
	public f32 targetValue;

	public func valid() -> bool {
		return lightIndex != skRegirSlotInvalid;
	}
};

public struct RegirCell {
	public RegirCellKey key;
	public f32v3 representativeOri;
	public f32v3 representativeNor;
	// normalization values, (1/M) * sum(ĥ / p)
	public f32 normalizationNoncanonical;
	public f32 normalizationCanonical;
	public u32 lastFillIndex;
	public u32 frameIndexLastUsed;
	// a shading point from this frame, becomes the representative next fill
	public f32v3 pendingOri;
	public f32v3 pendingNor;
};

} // namespace vkof
