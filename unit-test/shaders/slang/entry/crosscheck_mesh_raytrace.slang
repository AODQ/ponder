import "util-mesh-raytrace";
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

// util-mesh-raytrace's whole surface: the flat index -> meshlet -> vertex
// walk a ray hit resolves through, against a real mor model

// in:  primitiveIndex, barycentric(2), unpackDerivatives
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
	const u32 i = thread * 4u;
	const u32 o = thread * 20u;

	const u32 primitiveIndex = u32(inBuf[i + 0u]);
	const f32v2 barycentric = f32v2(inBuf[i + 1u], inBuf[i + 2u]);
	const bool unpackDerivatives = inBuf[i + 3u] > 0.5f;

	// glsl forbids const on a struct carrying a buffer reference
	UtilMeshAttributeDataFromIndices data = (
		utilMeshAttributeDataFromIndices(
			pc.modelsVa, 0u, primitiveIndex, barycentric, unpackDerivatives
		)
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
