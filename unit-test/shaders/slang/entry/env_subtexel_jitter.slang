import "util-random";
import "util-environment-map";

struct PC {
	u32v4 rootPad[8];
	u64 binVa;
	EnvironmentMapHandles handles;
	u32 sampleCount;
	u32 binCount;
	u32 seedSalt;
	f32 rotation;
};

[[vk::push_constant]] PC pc;

// sub-texel placement histogram for environmentMapImportanceSample.
//
// the alias draw picks a texel, then the *leftover precision* of the same
// two random numbers is reused as the continuous position inside it
// (util-environment-map.glsl:103-135). for u that leftover is
// `jitterX = xi.x * n - float(int(xi.x * n))` with n = width * height, so
// once n approaches f32's 24-bit mantissa the subtraction has no bits left
// to return and every sample collapses onto its texel's left edge -- where
// the bilinear radiance fetch returns the average of two adjacent texels
// rather than the one the pdf was read from.
//
// this histograms the fractional position each returned direction lands at
// within its own texel, in u and v separately. a correct sampler is uniform
// in both; the caller checks flatness and the exact-zero fraction.

// binVa layout: [0, bins) fractional-u histogram, [bins, 2*bins)
// fractional-v histogram, then two exact-zero counters (u, v) and a
// round-trip texel mismatch counter

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 sampleIndex = gl_GlobalInvocationID.x;
	if (sampleIndex >= pc.sampleCount) {
		return;
	}

	const u32 h0 = fnPcgHash(sampleIndex + fnPcgHash(pc.seedSalt));
	const u32 h1 = fnPcgHash(h0);
	u64 state = (u64(h0) << 32) | u64(h1);

	const f32v2 xi = f32v2(fnSampleUniform(state), fnSampleUniform(state));
	f32 pdf;
	const f32v3 dir = (
		environmentMapImportanceSample(pc.handles, xi, pc.rotation, pdf)
	);

	// back to the map's own uv, then to the position inside the texel
	const i32v2 size = textureSize(vkofTextures[pc.handles.pdf], 0);
	const f32v2 uv = environmentMapDirToUv(dir, pc.rotation);
	const f32 fracU = frac(uv.x * f32(size.x));
	const f32 fracV = frac(uv.y * f32(size.y));

	const u32 binU = min(u32(fracU * f32(pc.binCount)), pc.binCount - 1u);
	const u32 binV = min(u32(fracV * f32(pc.binCount)), pc.binCount - 1u);
	atomicAdd(&((u32 *)(pc.binVa))[binU], 1u);
	atomicAdd(&((u32 *)(pc.binVa))[pc.binCount + binV], 1u);

	// the collapse signature is an exact 0.0, not merely a small value
	if (fracU == 0.0f) {
		atomicAdd(&((u32 *)(pc.binVa))[2u * pc.binCount + 0u], 1u);
	}
	if (fracV == 0.0f) {
		atomicAdd(&((u32 *)(pc.binVa))[2u * pc.binCount + 1u], 1u);
	}

	// -- round trip: the texel the alias draw picked must be the texel
	// environmentMapPdf later reads its density from.
	//
	// the sampler hands back only a direction, and every consumer (the nee
	// weight, the mis weight, the pdf itself) recovers the texel by going
	// dir -> uv -> floor. so the returned pdf is only the pdf of the sample
	// if that round trip lands back on the texel that was drawn. when it
	// does not, a sample drawn from inside a sun is priced with a
	// neighbouring sky texel's density -- four orders of magnitude too
	// small -- and the estimator returns a correspondingly enormous value.
	//
	// the pick below must mirror environmentMapImportanceSample exactly
	const i32 n = size.x * size.y;
	const f32 nf = xi.x * f32(n);
	const i32 i = clamp(i32(nf), 0, n - 1);
	const f32 probThreshold = (
		texelFetch(
			vkofTextures[pc.handles.aliasProb],
			i32v2(i % size.x, i / size.x), 0
		).r
	);
	const i32 drawnIdx = (
		xi.y < probThreshold
			? i
			: i32(
				texelFetch(
					vkofTextures[pc.handles.aliasIndex],
					i32v2(i % size.x, i / size.x), 0
				).r + 0.5f
			)
	);
	const i32v2 drawnTexel = i32v2(drawnIdx % size.x, drawnIdx / size.x);
	const i32v2 roundTripTexel = i32v2(
		clamp(i32(uv.x * f32(size.x)), 0, size.x - 1),
		clamp(i32(uv.y * f32(size.y)), 0, size.y - 1)
	);
	if (any(drawnTexel != roundTripTexel)) {
		atomicAdd(&((u32 *)(pc.binVa))[2u * pc.binCount + 2u], 1u);
	}
}
