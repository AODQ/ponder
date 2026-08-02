import "util-fresnel";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
};

[[vk::push_constant]] PC pc;

// in:  cosTheta, eta, f0.rgb, specularWeight, specularColor.rgb
// out: dielectric, metallic.rgb
[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 thread = gl_GlobalInvocationID.x;
	if (thread >= pc.threadCount) { return; }

	f32 * inBuf = (f32 *)(pc.inVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 i = thread * 9u;
	const u32 o = thread * 4u;

	const f32 cosTheta = inBuf[i + 0u];
	const f32 eta = inBuf[i + 1u];
	const f32v3 f0 = f32v3(
		inBuf[i + 2u], inBuf[i + 3u], inBuf[i + 4u]
	);
	const f32 specularWeight = inBuf[i + 5u];
	const f32v3 specularColor = f32v3(
		inBuf[i + 6u], inBuf[i + 7u], inBuf[i + 8u]
	);

	const f32v3 metallic = utilFresnelMetallic(
		f0, specularWeight, specularColor, cosTheta
	);
	outBuf[o + 0u] = utilFresnelDielectric(cosTheta, eta);
	outBuf[o + 1u] = metallic.r;
	outBuf[o + 2u] = metallic.g;
	outBuf[o + 3u] = metallic.b;
}
