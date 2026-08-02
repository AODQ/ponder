import "util-environment-map";

struct PC {
	u32v4 rootPad[8];
	u64 outVa;
	EnvironmentMapHandles handles;
	u32 width;
	u32 height;
	f32 rotation;
};

[[vk::push_constant]] PC pc;

// dense nan/degenerate-value sweep over the environment map's sampling and
// lookup functions (modeled on ggx_bounded_sample_sweep.comp). one thread
// per (u,v) texel, two independent checks:
//  (a) environmentMapPdf/environmentMapRadiance evaluated directly at the
//      texel's own (u,v) -> dir (via environmentMapUvToDir) -- the only way
//      to force exact coverage of the seam (u=0/1) and poles (v=0/1),
//      which a stochastic xi draw essentially never lands on exactly
//  (b) environmentMapImportanceSample driven by an inner 8x8 xi substrata
//      grid anchored at this texel, so the full xi domain -- including the
//      exact xi=(0,0) corner the cdf binary search treats specially --
//      gets dense coverage every dispatch, not just at the outer grid's
//      own resolution

// 7 floats per texel: directBad (0/1), pdfDirect, sampleNonFiniteCount,
// sampleMaxUnitLengthError, sampleMinPdf, sampleMaxPdf, roundTripUvError

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32v2 texel = gl_GlobalInvocationID.xy;
	if (texel.x >= pc.width || texel.y >= pc.height) { return; }

	const f32 rotation = pc.rotation;

	// -- (a) direct (u,v) probe: closed [0,1] so both ends land exactly on
	// the seam / poles
	const f32v2 uvDirect = f32v2(
		f32(texel.x) / f32(pc.width - 1u),
		f32(texel.y) / f32(pc.height - 1u)
	);
	const f32v3 dirDirect = environmentMapUvToDir(uvDirect, rotation);
	const f32 pdfDirect = environmentMapPdf(pc.handles, dirDirect, rotation);
	const f32v3 radianceDirect = (
		environmentMapRadiance(pc.handles, dirDirect, rotation)
	);
	// pdf legitimately hits exactly 0 at the poles (vanishing solid angle,
	// see environmentMapPdf's sinTheta guard) -- >= 0 not > 0
	const bool directOk = (
		!isnan(pdfDirect) && !isinf(pdfDirect) && pdfDirect >= 0.0f
		&& !isnan(radianceDirect.x) && !isinf(radianceDirect.x)
		&& radianceDirect.x >= 0.0f
		&& !isnan(radianceDirect.y) && !isinf(radianceDirect.y)
		&& radianceDirect.y >= 0.0f
		&& !isnan(radianceDirect.z) && !isinf(radianceDirect.z)
		&& radianceDirect.z >= 0.0f
	);

	// -- uv -> dir -> uv round trip through the SAME rotation; a rotation
	// term applied inconsistently between environmentMapUvToDir and
	// environmentMapDirToUv would bias every rotated render (radiance looked
	// up at a different direction than the pdf claims). u is periodic, so
	// wrap-aware distance; at the poles every u maps to the same direction,
	// so only v is checked there
	const f32v2 uvRt = environmentMapDirToUv(dirDirect, rotation);
	const f32 sinThetaDirect = sin(uvDirect.y * PI);
	const f32 duRaw = abs(uvRt.x - uvDirect.x);
	const f32 du = min(duRaw, 1.0f - duRaw);
	const f32 dv = abs(uvRt.y - uvDirect.y);
	const f32 roundTripErr = (
		sinThetaDirect > 1e-3f ? max(du, dv) : dv
	);

	// -- (b) importance-sample sweep: half-open [0,1) base per texel so the
	// jittered xi never reaches exactly 1.0 (matching a real rng's [0,1)
	// contract), while texel (0,0)'s first substratum still hits the exact
	// xi=(0,0) domain corner
	const f32v2 xiBase = f32v2(
		f32(texel.x) / f32(pc.width), f32(texel.y) / f32(pc.height)
	);
	f32 nonFiniteCount = 0.0f;
	f32 maxLenErr = 0.0f;
	f32 minPdf = 1e30f;
	f32 maxPdf = 0.0f;
	const u32 kXiSteps = 8u;
	for (u32 xj = 0u; xj < kXiSteps; ++xj) {
		for (u32 xk = 0u; xk < kXiSteps; ++xk) {
			const f32v2 xi = xiBase + f32v2(
				f32(xj) / f32(kXiSteps) / f32(pc.width),
				f32(xk) / f32(kXiSteps) / f32(pc.height)
			);
			f32 pdf;
			const f32v3 dir = (
				environmentMapImportanceSample(pc.handles, xi, rotation, pdf)
			);
			const bool finite = (
				!isnan(dir.x) && !isnan(dir.y) && !isnan(dir.z)
				&& !isinf(dir.x) && !isinf(dir.y) && !isinf(dir.z)
				&& !isnan(pdf) && !isinf(pdf)
			);
			if (!finite) {
				nonFiniteCount += 1.0f;
				continue;
			}
			maxLenErr = max(maxLenErr, abs(length(dir) - 1.0f));
			minPdf = min(minPdf, pdf);
			maxPdf = max(maxPdf, pdf);
		}
	}

	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 base = (texel.y * pc.width + texel.x) * 7u;
	outBuf[base+0u] = directOk ? 0.0f : 1.0f;
	outBuf[base+1u] = pdfDirect;
	outBuf[base+2u] = nonFiniteCount;
	outBuf[base+3u] = maxLenErr;
	outBuf[base+4u] = minPdf;
	outBuf[base+5u] = maxPdf;
	outBuf[base+6u] = roundTripErr;
}
