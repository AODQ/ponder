import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-fuzz";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 outVa;
	u32 width;
	u32 height;
	f32 fuzzB;
};

[[vk::push_constant]] PC pc;

// dense parameter-space sweep hunting for NaN/Inf and degenerate output in
// the ported zeltner ltc fuzz sampler + pdf. one thread per (incidence x
// fuzzA) texel, fuzzB held fixed per-dispatch (see pc.fuzzB); each texel
// exercises an 8x8 xi grid that includes the exact 0 endpoint (the 1 - 2^-23
// upper corner is skipped, matching ggx_bounded_sample_sweep.comp's
// convention of carving boundary-flicker strata into their own test).
// openPbrFuzzSampleWo takes xi directly, so the grid drives the sampler
// with exact endpoint values.

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// texel (x, y) maps to cosThetaI in [-1, 1] over x (below-horizon and
// exact-antipodal wi included: utilFuzzFrame has a dedicated degenerate
// fallback) and fuzzA in [1e-5, 3] over y; fuzzB is a fixed per-dispatch
// constant. output 4 floats per texel: nonFiniteCount, maxUnitLengthError,
// minPdf, maxPdf.

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32v2 texel = gl_GlobalInvocationID.xy;
	if (texel.x >= pc.width || texel.y >= pc.height) { return; }

	const f32 cosThetaI = (
		-1.0f + 2.0f * f32(texel.x) / f32(pc.width - 1u)
	);
	const f32 sinThetaI = sqrt(max(0.0f, 1.0f - cosThetaI * cosThetaI));
	const f32v3 wi = f32v3(sinThetaI, 0.0f, cosThetaI);
	const f32 fuzzA = (
		1e-5f + 3.0f * f32(texel.y) / f32(pc.height - 1u)
	);
	const f32 fuzzB = pc.fuzzB;

	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);

	f32 nonFiniteCount = 0.0f;
	f32 maxLenErr = 0.0f;
	f32 minPdf = 1e30f;
	f32 maxPdf = -1e30f;

	const u32 kXiSteps = 8u;
	for (u32 xj = 0u; xj < kXiSteps; ++xj) {
		for (u32 xk = 0u; xk < kXiSteps; ++xk) {
			const f32v2 xi = f32v2(
				f32(xj) / f32(kXiSteps), f32(xk) / f32(kXiSteps)
			);
			const f32v3 wo = openPbrFuzzSampleWo(nor, wi, fuzzA, fuzzB, xi);
			const f32 pdf = openPbrFuzzPdf(nor, wi, wo, fuzzA, fuzzB);

			const bool finite = (
				!isnan(wo.x) && !isnan(wo.y) && !isnan(wo.z)
				&& !isinf(wo.x) && !isinf(wo.y) && !isinf(wo.z)
				&& !isnan(pdf) && !isinf(pdf)
			);
			if (!finite) {
				nonFiniteCount += 1.0f;
				continue;
			}
			maxLenErr = max(maxLenErr, abs(length(wo) - 1.0f));
			minPdf = min(minPdf, pdf);
			maxPdf = max(maxPdf, pdf);
		}
	}

	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 base = (texel.y * pc.width + texel.x) * 4u;
	outBuf[base+0u] = nonFiniteCount;
	outBuf[base+1u] = maxLenErr;
	outBuf[base+2u] = minPdf;
	outBuf[base+3u] = maxPdf;
}
