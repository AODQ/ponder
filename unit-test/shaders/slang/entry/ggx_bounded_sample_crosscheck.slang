import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
#include "../util-microfacet-materialx-reference.slang"
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// cross-check of the bounded ggx sampler/pdf against materialx's
// spherical-cap vndf sampler and pdf. the two samplers share the identical
// cap parameterization (phi = 2 pi xi.x, z = (1 - xi.y)(1 + b) - b); they
// differ only in the cap's lower bound b. for wi.z <= 0 the bounded sampler
// falls back to b = iStd.z, which is exactly materialx's unbounded cap, so
// the sampled directions must match bit-for-bit given the same xi. for
// wi.z > 0 the bound k shrinks the cap and only the pdfs remain related:
// ponderPdf / mxPdf = (iz + t) / (k iz + t), asserted host-side.

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the lobe-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// per sample, 7 floats in: wi.xyz, alpha.xy, xi.xy. output 8 floats:
// mineWo.xyz, refWo.xyz, minePdf, refPdf. canonical local frame
// (tanX=(1,0,0), tanY=(0,1,0), nor=(0,0,1)).
//
// refWo reflects wi about materialx's sampled half-vector; only comparable
// to mineWo when wi.z <= 0 (see header). refPdf uses materialx's
// mx_ggx_VNDF_reflection_PDF for wi.z > 0 (isotropic alpha only: materialx's
// G1 is scalar-alpha); for wi.z <= 0 the G1-based form is out of its domain,
// so the unbounded-cap pdf D / (2 (iz + t)) is assembled from mx_ggx_NDF
// directly.

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 7u;
	const f32v3 wi = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32v2 alpha = f32v2(inBuf[base+3u], inBuf[base+4u]);
	const f32v2 xi = f32v2(inBuf[base+5u], inBuf[base+6u]);

	const f32v3 tanX = f32v3(1.0f, 0.0f, 0.0f);
	const f32v3 tanY = f32v3(0.0f, 1.0f, 0.0f);
	const f32v3 nor = f32v3(0.0f, 0.0f, 1.0f);

	const f32v3 mineWo = (
		utilMicrofacetSampleGgxBoundedWoAniso(tanX, tanY, nor, wi, alpha, xi)
	);
	const f32 minePdf = (
		utilMicrofacetGgxBoundedReflectPdfAniso(
			tanX, tanY, nor, wi, mineWo, alpha
		)
	);

	const f32v3 refH = mx_ggx_importance_sample_VNDF(xi, wi, alpha);
	const f32v3 refWo = 2.0f * dot(wi, refH) * refH - wi;

	// pdf of the direction the ponder sampler produced, from materialx
	// pieces; evaluated at mineWo's half-vector so both pdfs describe the
	// same direction
	const f32v3 h = normalize(wi + mineWo);
	const f32 t = (
		length(f32v3(alpha.x * wi.x, alpha.y * wi.y, wi.z))
	);
	f32 refPdf;
	if (wi.z > 0.0f) {
		refPdf = (
			mx_ggx_VNDF_reflection_PDF(
				h, alpha, mx_ggx_smith_G1(wi.z, alpha.x), wi.z
			)
		);
	} else {
		refPdf = mx_ggx_NDF(h, alpha) / (2.0f * (wi.z + t));
	}

	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[i*8u+0u] = mineWo.x;
	outBuf[i*8u+1u] = mineWo.y;
	outBuf[i*8u+2u] = mineWo.z;
	outBuf[i*8u+3u] = refWo.x;
	outBuf[i*8u+4u] = refWo.y;
	outBuf[i*8u+5u] = refWo.z;
	outBuf[i*8u+6u] = minePdf;
	outBuf[i*8u+7u] = refPdf;
}
