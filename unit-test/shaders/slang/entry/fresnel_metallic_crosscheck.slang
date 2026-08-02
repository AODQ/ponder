import "util-fresnel";
#include "../util-fresnel-materialx-reference.slang"

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 mineOutVa;
	u64 refOutVa;
	u32 count;
};

[[vk::push_constant]] PC pc;

// per sample, 8 floats: f0.xyz, specularWeight, specularColor.xyz, dotNorWi

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 i = gl_GlobalInvocationID.x;
	if (i >= pc.count) { return; }

	f32 * inBuf = (f32 *)(pc.inVa);
	u32 base = i * 8u;
	const f32v3 f0 = (
		f32v3(inBuf[base+0u], inBuf[base+1u], inBuf[base+2u])
	);
	const f32 specularWeight = inBuf[base+3u];
	const f32v3 specularColor = (
		f32v3(inBuf[base+4u], inBuf[base+5u], inBuf[base+6u])
	);
	const f32 dotNorWi = inBuf[base+7u];

	((f32v3 *)(pc.mineOutVa))[i] = (
		utilFresnelMetallic(f0, specularWeight, specularColor, dotNorWi)
	);
	// clamped to match ponder's deliberate f82 >= 0 deviation (see
	// utilFresnelMetallic); materialx's own hoffman-schlick returns the
	// raw, possibly-negative value
	((f32v3 *)(pc.refOutVa))[i] = (
		max(
			mxFresnelHoffmanSchlick(
				dotNorWi, f0, specularWeight * specularColor, f32v3(1.0f), 5.0f
			),
			f32v3(0.0f)
		)
	);
}
