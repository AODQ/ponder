import "util-material-openpbr-tables";
import "util-material-openpbr-thinfilm";
import "util-material-openpbr-microfacet";
import "util-material-openpbr-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
};

[[vk::push_constant]] PC pc;

// the header expects its includer to supply these, as
// util-material-openpbr.glsl does

// in:  h.xyz, alpha.xy, wi.xyz, wo.xyz, eta, xi.xy
// out: D, Daniso, G1, Vaniso, fresnel, vndf.xyz, pdf, halfValid
[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 thread = gl_GlobalInvocationID.x;
	if (thread >= pc.threadCount) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 i = thread * 13u;
	const u32 o = thread * 10u;

	const f32v3 h = normalize(f32v3(inBuf[i+0u], inBuf[i+1u], abs(inBuf[i+2u]) + 1e-3f));
	const f32v2 alpha = f32v2(inBuf[i+3u], inBuf[i+4u]);
	const f32v3 wi = normalize(f32v3(inBuf[i+5u], inBuf[i+6u], abs(inBuf[i+7u]) + 1e-3f));
	const f32v3 wo = normalize(f32v3(inBuf[i+8u], inBuf[i+9u], abs(inBuf[i+10u]) + 1e-3f));
	const f32 eta = inBuf[i+11u];
	const f32v2 xi = f32v2(inBuf[i+12u], 1.0f - inBuf[i+12u]);

	f32v3 hOut;
	const bool halfValid = utilHalfVectorLocal(wi, wo, hOut);
	const f32v3 vndf = utilMicrofacetSampleGgxVndfAniso(
		f32v3(1,0,0), f32v3(0,1,0), f32v3(0,0,1), wi, alpha, xi
	);

	outBuf[o+0u] = utilMicrofacetGgxDistribution(h.z, alpha.x);
	outBuf[o+1u] = utilMicrofacetGgxDistributionAniso(h, alpha);
	outBuf[o+2u] = utilMicrofacetSmithG1(wi.z, alpha.x);
	outBuf[o+3u] = utilMicrofacetSmithGgxVisibilityAniso(wi, wo, alpha);
	outBuf[o+4u] = utilMicrofacetFresnelDielectric(h.z, eta);
	outBuf[o+5u] = vndf.x;
	outBuf[o+6u] = vndf.y;
	outBuf[o+7u] = vndf.z;
	outBuf[o+8u] = utilMicrofacetGgxBoundedReflectPdfAniso(
		f32v3(1,0,0), f32v3(0,1,0), f32v3(0,0,1), wi, wo, alpha
	);
	outBuf[o+9u] = halfValid ? 1.0f : 0.0f;
}
