import "util-mesh";
import "util-mor-shared";
import "util-resolve-pc";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
	u64 modelsVa;
	u64 unusedVa;
};

[[vk::push_constant]] PC pc;

#define RESOLVE_NO_PC

// utilMeshDecodeAttributes, the visibility-buffer resolve path, against a
// real mor model. the texel is derived from a primitive index through the
// model's own flat meshlet table, so every (meshletId, triangleId) pair the
// shader sees is one the model actually has

// in:  primitiveIndex, pixel(2), screensize(2)
// out: materialIndex, origin(3), uv(2), uvDx(2), uvDy(2), normal(3),
//      normalGeometrical(3), tangent(4)
[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 thread = gl_GlobalInvocationID.x;
	if (thread >= pc.threadCount) { return; }

	f32 * inBuf = (f32 *)(pc.inVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 i = thread * 5u;
	const u32 o = thread * 20u;

	GpuResolveModelIndirect * modelsBuf = (
		(GpuResolveModelIndirect *)(pc.modelsVa)
	);
	const GpuResolveModelIndirect model = modelsBuf[0];

	const u32 primitiveIndex = u32(inBuf[i + 0u]);
	u32 * flatMeshlets = (
		(u32 *)(model.flatMeshlets)
	);
	const u32 meshletId = flatMeshlets[primitiveIndex];
	// the flat table is ordered by meshlet, so walking back to the first
	// primitive of this meshlet gives the triangle's index within it
	u32 triangleId = 0u;
	for (u32 k = 0u; k < 128u; ++k) {
		if (primitiveIndex < k + 1u) { break; }
		if (flatMeshlets[primitiveIndex - k - 1u] != meshletId) { break; }
		triangleId = k + 1u;
	}

	UtilMeshUnpacked texelUnpacked;
	texelUnpacked.modelId = 0u;
	texelUnpacked.meshletId = meshletId;
	texelUnpacked.triangleId = triangleId;

	UtilMeshDecodeContext ctx;
	ctx.models = pc.modelsVa;
	ctx.viewProj = f32m44(
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
	);
	ctx.invocationId = u32v2(
		u32(inBuf[i + 1u]), u32(inBuf[i + 2u])
	);
	const i32v2 screensize = i32v2(
		i32(inBuf[i + 3u]), i32(inBuf[i + 4u])
	);

	UtilMeshAttributeData data = utilMeshDecodeAttributes(
		ctx, texelUnpacked, screensize, true, true, true, true, true
	);

	outBuf[o + 0u] = f32(data.materialIndex);
	outBuf[o + 1u] = data.origin.x;
	outBuf[o + 2u] = data.origin.y;
	outBuf[o + 3u] = data.origin.z;
	outBuf[o + 4u] = data.uv.x;
	outBuf[o + 5u] = data.uv.y;
	outBuf[o + 6u] = data.uvDx.x;
	outBuf[o + 7u] = data.uvDx.y;
	outBuf[o + 8u] = data.uvDy.x;
	outBuf[o + 9u] = data.uvDy.y;
	outBuf[o + 10u] = data.normal.x;
	outBuf[o + 11u] = data.normal.y;
	outBuf[o + 12u] = data.normal.z;
	outBuf[o + 13u] = data.normalGeometrical.x;
	outBuf[o + 14u] = data.normalGeometrical.y;
	outBuf[o + 15u] = data.normalGeometrical.z;
	outBuf[o + 16u] = data.tangent.x;
	outBuf[o + 17u] = data.tangent.y;
	outBuf[o + 18u] = data.tangent.z;
	outBuf[o + 19u] = data.tangent.w;
}
