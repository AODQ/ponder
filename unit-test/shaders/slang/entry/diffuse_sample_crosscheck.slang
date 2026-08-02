import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-diffuse";
#include "../util-microfacet-materialx-reference.slang"
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// cross-check of the cosine-hemisphere sampler and pdf against materialx.
// ponder maps (u.x -> cosTheta = sqrt(u.x), u.y -> phi = TAU u.y) while
// materialx maps (Xi.x -> phi, Xi.y -> cosTheta = sqrt(Xi.y)); feeding
// materialx the swizzled Xi = xi.yx makes both draw the identical local
// direction, so the host can assert direct vector equality.

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the diffuse-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// per sample, 2 floats in: xi.xy (the raw uniform pair; this shader feeds
// the sampler's core mapping -- normalize(utilToCartesian(sqrt(u.x),
// TAU u.y)) -- directly, exactly as utilCosineHemisphereSampleWo composes
// it, keeping the comparison frame-independent. the reorientation half is
// covered by the shading-frame suite, the full drawn path by the
// chi-square test). output 8 floats: mineLocal.xyz, mxLocal.xyz,
// mxPdfAtMineZ, minePdfAtMineZ.

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	const f32v2 xi = f32v2(inBuf[i*2u+0u], inBuf[i*2u+1u]);

	// the sampler's core mapping, exactly as utilCosineHemisphereSampleWo
	// composes it from the drawn pair
	const f32v3 mineLocal = (
		normalize(utilToCartesian(sqrt(xi.x), TAU * xi.y))
	);
	// materialx's identical mapping under the documented xi swizzle
	const f32v3 mxLocal = mx_cosine_sample_hemisphere(xi.yx);

	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[i*8u+0u] = mineLocal.x;
	outBuf[i*8u+1u] = mineLocal.y;
	outBuf[i*8u+2u] = mineLocal.z;
	outBuf[i*8u+3u] = mxLocal.x;
	outBuf[i*8u+4u] = mxLocal.y;
	outBuf[i*8u+5u] = mxLocal.z;
	outBuf[i*8u+6u] = mx_cosine_hemisphere_PDF(mineLocal.z);
	outBuf[i*8u+7u] = openPbrGlossyDiffusePdf(mineLocal.z);
}
