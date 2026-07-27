// util-mesh's model-unpack from a ray hit's (drawIndex, primitiveIndex) --
// split out from util-mesh.glsl since it needs GpuFlatIndexBuffer/
// GpuFlatMeshletBuffer (a flattened triangle->meshlet index the raytracer
// builds for O(1) hit resolution), which GpuResolveModelIndirect only
// carries in apps that actually raytrace -- the visibility-buffer resolve
// path (util-mesh.glsl's utilMeshDecodeAttributes) doesn't need it

#ifndef UTIL_MESH_RAYTRACE_GLSL
#define UTIL_MESH_RAYTRACE_GLSL

struct UtilMeshAttributeDataFromIndices {
	GpuMorMaterialBuffer materialBuf;
	uint materialIndex;
	GpuMorUvTransformBuffer uvTransforms;
	f32v3 origin;
	f32v2 uv;
	f32v2 uvDx;
	f32v2 uvDy;
	f32v3 normal;
	f32v3 normalGeometrical;
	f32v4 tangent;
};

UtilMeshAttributeDataFromIndices utilMeshAttributeDataFromIndices(
	const uint modelDrawIndex,
	const uint primitiveIndex,
	const vec2 barycentric,
	const bool unpackUvDerivatives
) {
	GpuResolveModelIndirectBuffer modelsBuf = (
		GpuResolveModelIndirectBuffer(pc.global.models)
	);
	const GpuResolveModelIndirect model = modelsBuf.data[modelDrawIndex];

	GpuFlatIndexBuffer flatIdxBuf = GpuFlatIndexBuffer(model.flatIndices);
	const uint i0 = flatIdxBuf.data[primitiveIndex * 3u + 0u];
	const uint i1 = flatIdxBuf.data[primitiveIndex * 3u + 1u];
	const uint i2 = flatIdxBuf.data[primitiveIndex * 3u + 2u];

	GpuMorPositionBuffer positionBuf = (
		GpuMorPositionBuffer(model.positions)
	);
	const f32v3 pos0 = positionBuf.data[i0];
	const f32v3 pos1 = positionBuf.data[i1];
	const f32v3 pos2 = positionBuf.data[i2];
	GpuMorVertexAttributeBuffer attrBuf = (
		GpuMorVertexAttributeBuffer(model.attributes)
	);
	const GpuMorVertexAttribute attr0 = attrBuf.data[i0];
	const GpuMorVertexAttribute attr1 = attrBuf.data[i1];
	const GpuMorVertexAttribute attr2 = attrBuf.data[i2];

	GpuMorMeshletBuffer meshletBuf = (
		GpuMorMeshletBuffer(model.meshlets)
	);
	const uint meshletId = (
		GpuFlatMeshletBuffer(model.flatMeshlets).data[primitiveIndex]
	);
	const GpuMorMeshlet meshlet = meshletBuf.data[meshletId];

	UtilMeshAttributeDataFromIndices retAttrData;
	retAttrData.materialBuf = GpuMorMaterialBuffer(model.materials);
	retAttrData.materialIndex = meshlet.materialIndex;
	retAttrData.uvTransforms = GpuMorUvTransformBuffer(model.uvTransforms);
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
	const mat3 modelRot = mat3(model.modelMatrix);
	const mat3 normalMat = transpose(inverse(modelRot));
	const f32v3 worldNormal = normalMat * localNormal;
	retAttrData.normal = (
		dot(worldNormal, worldNormal) > 1e-12f
		? normalize(worldNormal)
		: f32v3(0.0f)
	);
	retAttrData.normalGeometrical = (
		normalize(normalMat * cross(pos1 - pos0, pos2 - pos0))
	);
	const f32v3 worldTangent = modelRot * localTangent.xyz;
	retAttrData.tangent = f32v4(
		dot(worldTangent, worldTangent) > 1e-12f
		? normalize(worldTangent)
		: f32v3(0.0f),
		localTangent.w
	);
	return retAttrData;
}

#endif // UTIL_MESH_RAYTRACE_GLSL
