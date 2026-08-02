import "util-shading-frame";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
};

[[vk::push_constant]] PC pc;

// in:  nor.xyz, tangent.xyz, tangent.w, rotation, xi.xy
// out: fromNormal.tanX.xyz, fromTangent.tanX.xyz, rotated.tanY.xyz,
//      cosineSample.xyz, tbn[0].xyz
[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 thread = gl_GlobalInvocationID.x;
	if (thread >= pc.threadCount) { return; }

	f32 * inBuf = (f32 *)(pc.inVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 i = thread * 10u;
	const u32 o = thread * 15u;

	const f32v3 nor = normalize(f32v3(
		inBuf[i + 0u], inBuf[i + 1u], inBuf[i + 2u]
	));
	const f32v3 tangent = f32v3(
		inBuf[i + 3u], inBuf[i + 4u], inBuf[i + 5u]
	);
	const f32 tangentW = inBuf[i + 6u];
	const f32 rotation = inBuf[i + 7u];
	const f32v2 xi = f32v2(inBuf[i + 8u], inBuf[i + 9u]);

	const ShadingFrame fromNormal = shadingFrameFromNormal(nor);
	const ShadingFrame fromTangent = shadingFrameFromTangent(nor, tangent);
	const ShadingFrame rotated = shadingFrameRotate(fromNormal, rotation);
	const f32v3 cosineSample = utilCosineHemisphereSampleWo(nor, xi);
	const f32m3x3 tbn = utilCalculateTbnBasis(nor, f32v4(tangent, tangentW));

	outBuf[o + 0u] = fromNormal.tanX.x;
	outBuf[o + 1u] = fromNormal.tanX.y;
	outBuf[o + 2u] = fromNormal.tanX.z;
	outBuf[o + 3u] = fromTangent.tanX.x;
	outBuf[o + 4u] = fromTangent.tanX.y;
	outBuf[o + 5u] = fromTangent.tanX.z;
	outBuf[o + 6u] = rotated.tanY.x;
	outBuf[o + 7u] = rotated.tanY.y;
	outBuf[o + 8u] = rotated.tanY.z;
	outBuf[o + 9u] = cosineSample.x;
	outBuf[o + 10u] = cosineSample.y;
	outBuf[o + 11u] = cosineSample.z;
	outBuf[o + 12u] = tbn[0].x;
	outBuf[o + 13u] = tbn[0].y;
	outBuf[o + 14u] = tbn[0].z;
}
