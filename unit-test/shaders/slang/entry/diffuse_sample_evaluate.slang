import "util-random";
import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-diffuse";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// constants normally provided by util-material-openpbr.glsl; the full
// material include chain pulls in every lobe, so the diffuse-only prelude
// is replicated here instead (values must match util-material-openpbr.glsl)
static const f32 skFon1 = 0.5f - 2.0f / (3.0f * PI);
static const f32 skFon2 = 2.0f / 3.0f - 28.0f / (15.0f * PI);

// per sample, 5 floats in: nor.xyz, xi.xy. output 10 floats: woGlossy.xyz
// (openPbrGlossyDiffuseSampleWo), pdf at woGlossy
// (openPbrGlossyDiffusePdf(dot(nor, woGlossy))), pad.xy (kept so the
// 10-float layout matches the harness), woCosine.xyz
// (utilCosineHemisphereSampleWo from the same xi; must equal woGlossy
// since the glossy sampler is a pass-through), pad.

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 5u;
	const f32v3 nor = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32v2 xi = f32v2(inBuf[base+3u], inBuf[base+4u]);

	const f32v3 woGlossy = openPbrGlossyDiffuseSampleWo(nor, xi);
	const f32 pdf = openPbrGlossyDiffusePdf(dot(normalize(nor), woGlossy));

	const f32v3 woCosine = utilCosineHemisphereSampleWo(nor, xi);

	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[i*10u+0u] = woGlossy.x;
	outBuf[i*10u+1u] = woGlossy.y;
	outBuf[i*10u+2u] = woGlossy.z;
	outBuf[i*10u+3u] = pdf;
	outBuf[i*10u+4u] = 0.0f;
	outBuf[i*10u+5u] = 0.0f;
	outBuf[i*10u+6u] = woCosine.x;
	outBuf[i*10u+7u] = woCosine.y;
	outBuf[i*10u+8u] = woCosine.z;
	outBuf[i*10u+9u] = 0.0f;
}
