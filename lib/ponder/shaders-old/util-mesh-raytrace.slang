module "util-mesh-raytrace";
__exported import "util-types";
__exported import "util-mor-shared";
__exported import "util-resolve-pc";
// util-mesh's model-unpack from a ray hit's (drawIndex, primitiveIndex) --
// split out from util-mesh.glsl since it needs GpuFlatIndexBuffer/
// GpuFlatMeshletBuffer (a flattened triangle->meshlet index the raytracer
// builds for O(1) hit resolution), which GpuResolveModelIndirect only
// carries in apps that actually raytrace -- the visibility-buffer resolve
// path (util-mesh.glsl's utilMeshDecodeAttributes) doesn't need it


public struct UtilMeshAttributeDataFromIndices {
	public GpuMorMaterial * materialBuf;
	public u32 materialIndex;
	public GpuMorUvTransform * uvTransforms;
	public f32v3 origin;
	public f32v2 uv;
	public f32v2 uvDx;
	public f32v2 uvDy;
	public f32v3 normal;
	public f32v3 normalGeometrical;
	public f32v4 tangent;
};

public func utilMeshAttributeDataFromIndices(
	models: u64,
	modelDrawIndex: u32,
	primitiveIndex: u32,
	barycentric: f32v2,
	unpackUvDerivatives: bool,
) -> UtilMeshAttributeDataFromIndices {
	GpuResolveModelIndirect * modelsBuf = (
		(GpuResolveModelIndirect *)(models)
	);
	const GpuResolveModelIndirect model = modelsBuf[modelDrawIndex];

	u32 * flatIdxBuf = (u32 *)(model.flatIndices);
	const u32 i0 = flatIdxBuf[primitiveIndex * 3u + 0u];
	const u32 i1 = flatIdxBuf[primitiveIndex * 3u + 1u];
	const u32 i2 = flatIdxBuf[primitiveIndex * 3u + 2u];

	f32v3 * positionBuf = (
		(f32v3 *)(model.positions)
	);
	const f32v3 pos0 = positionBuf[i0];
	const f32v3 pos1 = positionBuf[i1];
	const f32v3 pos2 = positionBuf[i2];
	GpuMorVertexAttribute * attrBuf = (
		(GpuMorVertexAttribute *)(model.attributes)
	);
	const GpuMorVertexAttribute attr0 = attrBuf[i0];
	const GpuMorVertexAttribute attr1 = attrBuf[i1];
	const GpuMorVertexAttribute attr2 = attrBuf[i2];

	GpuMorMeshlet * meshletBuf = (
		(GpuMorMeshlet *)(model.meshlets)
	);
	const u32 meshletId = (
		((u32 *)(model.flatMeshlets))[primitiveIndex]
	);
	const GpuMorMeshlet meshlet = meshletBuf[meshletId];

	UtilMeshAttributeDataFromIndices retAttrData;
	retAttrData.materialBuf = (GpuMorMaterial *)(model.materials);
	retAttrData.materialIndex = meshlet.materialIndex;
	retAttrData.uvTransforms = (GpuMorUvTransform *)(model.uvTransforms);
	retAttrData.uv = (
		attr0.uv * (1.0 - barycentric.x - barycentric.y)
		+ attr1.uv * barycentric.x
		+ attr2.uv * barycentric.y
	);
	retAttrData.uvDx = f32v2(0.0f);
	retAttrData.uvDy = f32v2(0.0f);
	// unpack uv derivatives TODO - ray differential? not sure

	const f32v3 localNormal = (
		attr0.normal * (1.0 - barycentric.x - barycentric.y)
		+ attr1.normal * barycentric.x
		+ attr2.normal * barycentric.y
	);
	const f32v4 localTangent = (
		attr0.tangent * (1.0 - barycentric.x - barycentric.y)
		+ attr1.tangent * barycentric.x
		+ attr2.tangent * barycentric.y
	);
	// attributes are model-local; the traced geometry is transformed by the
	// tlas instance transform
	const f32m3x3 modelRot = f32m3x3(model.modelMatrix);
	const f32m3x3 normalMat = transpose(inverse(modelRot));
	const f32v3 worldNormal = mul(normalMat, localNormal);
	retAttrData.normal = (
		dot(worldNormal, worldNormal) > 1e-12f
		? normalize(worldNormal)
		: f32v3(0.0f)
	);
	retAttrData.normalGeometrical = (
		normalize(mul(normalMat, cross(pos1 - pos0, pos2 - pos0)))
	);
	const f32v3 worldTangent = mul(modelRot, localTangent.xyz);
	retAttrData.tangent = f32v4(
		dot(worldTangent, worldTangent) > 1e-12f
		? normalize(worldTangent)
		: f32v3(0.0f),
		localTangent.w
	);
	return retAttrData;
}

