import "util-mesh";
import "util-mor-shared";
import "util-resolve-pc";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
};

[[vk::push_constant]] PC pc;

// util-mesh's decode path reaches the model table; this file only exercises
// the pure half, but the include has to resolve. RESOLVE_NO_PC keeps the
// header's own push constant out of the way of this shader's
#define RESOLVE_NO_PC

// the pure half of util-mesh: the visibility-texel bit layout, the 2d
// barycentric solve and the ndc mapping. utilMeshDecodeAttributes needs a
// real mor model and is covered separately

// in:  packed, p(2), a(2), b(2), c(2), ndc(4), screen(2)
// out: triangleId, meshletId, modelId, bary(3), denom, clip(2)
[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 thread = gl_GlobalInvocationID.x;
	if (thread >= pc.threadCount) { return; }

	f32 * inBuf = (f32 *)(pc.inVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 i = thread * 15u;
	const u32 o = thread * 9u;

	const UtilMeshUnpacked unpacked = (
		utilMeshUnpackTexel(asuint(inBuf[i + 0u]))
	);
	outBuf[o + 0u] = f32(unpacked.triangleId);
	outBuf[o + 1u] = f32(unpacked.meshletId);
	outBuf[o + 2u] = f32(unpacked.modelId);

	f32 denom = 0.0f;
	const f32v3 bary = utilBarycentric(
		f32v2(inBuf[i + 1u], inBuf[i + 2u]),
		f32v2(inBuf[i + 3u], inBuf[i + 4u]),
		f32v2(inBuf[i + 5u], inBuf[i + 6u]),
		f32v2(inBuf[i + 7u], inBuf[i + 8u]),
		denom
	);
	outBuf[o + 3u] = bary.x;
	outBuf[o + 4u] = bary.y;
	outBuf[o + 5u] = bary.z;
	outBuf[o + 6u] = denom;

	const f32v2 clip = utilNdcToClip(
		f32v4(
			inBuf[i + 9u], inBuf[i + 10u],
			inBuf[i + 11u], inBuf[i + 12u]
		),
		i32v2(i32(inBuf[i + 13u]), i32(inBuf[i + 14u]))
	);
	outBuf[o + 7u] = clip.x;
	outBuf[o + 8u] = clip.y;
}
