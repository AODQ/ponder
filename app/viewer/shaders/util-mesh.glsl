// util to pack and unpack mesh data

#ifndef UTIL_MESH_GLSL
#define UTIL_MESH_GLSL

// -----------------------------------------------------------------------------
// -- 32-bit packing of mesh data
// -----------------------------------------------------------------------------

struct UtilMeshUnpacked {
	uint triangleId;
	uint meshletId;
	uint modelId;
};

UtilMeshUnpacked utilMeshUnpackTexel(const uint packed) {
	UtilMeshUnpacked unpacked;
	// mash is: modelId: 8 bits, meshletId: 17 bits, triangleId: 7 bits
	// meshlet is offset by 1 to allow 0 to be reserved for "no meshlet"
	unpacked.triangleId = packed & 0x7Fu;
	unpacked.meshletId = ((packed >> 7u) & 0x1FFFFu) - 1u;
	unpacked.modelId = (packed >> 24u) & 0xFFu;
	return unpacked;
}

// -----------------------------------------------------------------------------
// -- encoding of attributes in meshlet data
// -----------------------------------------------------------------------------

struct UtilMeshAttributeData {
	GpuMorMaterial material;
	uint materialIndex;
	GpuMorUvTransformBuffer uvTransforms;
	f32v3 origin;
	f32v2 uv;
	f32v2 uvDx;
	f32v2 uvDy;
	f32v3 normal;
	f32v4 tangent;
	f32v3 normalGeometrical;
};

f32v3 utilBarycentric(
	const f32v2 p,
	const f32v2 a,
	const f32v2 b,
	const f32v2 c,
	out f32 outDenom
) {
	const f32v2 ra = a - p;
	const f32v2 rb = b - p;
	const f32v2 rc = c - p;
	outDenom = (
		  ra.x * (rb.y - rc.y)
		+ rb.x * (rc.y - ra.y)
		+ rc.x * (ra.y - rb.y)
	);
	const f32 inv = 1.0f / outDenom;
	const f32 u = (rb.x * rc.y - rc.x * rb.y) * inv;
	const f32 v = (rc.x * ra.y - ra.x * rc.y) * inv;
	return vec3(u, v, 1.0f - u - v);
}

// ndc to clip is in vulkan conventions
f32v2 utilNdcToClip(f32v4 ndc, const ivec2 screensize) {
	const f32v2 screen = ndc.xy / ndc.w * 0.5f + 0.5f;
	return screen * f32v2(screensize);
}

UtilMeshAttributeData utilMeshDecodeAttributes(
	const UtilMeshUnpacked texelUnpacked,
	const bool unpackWorldOrigin,
	const bool unpackNormal,
	const bool unpackUv,
	const bool unpackUvDerivatives,
	const bool unpackMaterial
) {
	const uint modelId = texelUnpacked.modelId;
	const uint meshletId = texelUnpacked.meshletId;
	const uint triangleId = texelUnpacked.triangleId;

	UtilMeshAttributeData retAttrData;

	// -- unpack model/meshlet/triangle data
	GpuResolveModelIndirectBuffer modelsBuf = (
		GpuResolveModelIndirectBuffer(pc.global.models)
	);
	const GpuResolveModelIndirect modelDesc = modelsBuf.data[modelId];
	const GpuMorMeshlet meshlet = (
		GpuMorMeshletBuffer(modelDesc.meshlets).data[meshletId]
	);
	GpuMorMeshletTriBuffer meshletTriBuffer = (
		GpuMorMeshletTriBuffer(modelDesc.meshletTris)
	);
	GpuMorMeshletVertBuffer meshletVertBuffer = (
		GpuMorMeshletVertBuffer(modelDesc.meshletVerts)
	);
	const GpuMorInstance instance = (
		GpuMorInstanceBuffer(modelDesc.instances).data[meshlet.instanceIndex]
	);

	// -- unpack triangle vertex indices
	const uint triangleBaseIdx = meshlet.triangleOffset + triangleId * 3;
	const uint localIdx0 = uint(meshletTriBuffer.data[triangleBaseIdx + 0]);
	const uint localIdx1 = uint(meshletTriBuffer.data[triangleBaseIdx + 1]);
	const uint localIdx2 = uint(meshletTriBuffer.data[triangleBaseIdx + 2]);

	// -- convert local vertex indices to global vertex indices
	const uint globalIdx0 = (
		meshletVertBuffer.data[meshlet.vertexOffset + localIdx0]
	);
	const uint globalIdx1 = (
		meshletVertBuffer.data[meshlet.vertexOffset + localIdx1]
	);
	const uint globalIdx2 = (
		meshletVertBuffer.data[meshlet.vertexOffset + localIdx2]
	);

	// -- unpack vertex positions
	GpuMorPositionBuffer positionBuf = (
		GpuMorPositionBuffer(modelDesc.positions)
	);
	const f32v3 pos0 = positionBuf.data[globalIdx0];
	const f32v3 pos1 = positionBuf.data[globalIdx1];
	const f32v3 pos2 = positionBuf.data[globalIdx2];

	// -- unpack attributes
	GpuMorVertexAttributeBuffer attrBuf = (
		GpuMorVertexAttributeBuffer(modelDesc.attributes)
	);
	const GpuMorVertexAttribute attr0 = attrBuf.data[globalIdx0];
	const GpuMorVertexAttribute attr1 = attrBuf.data[globalIdx1];
	const GpuMorVertexAttribute attr2 = attrBuf.data[globalIdx2];

	// -- reconstruct clip space
	const f32m44 modelTransform = (
		modelDesc.modelMatrix * instance.transform
	);
	const f32v4 ndc0 = pc.global.viewProj * modelTransform * f32v4(pos0, 1.0);
	const f32v4 ndc1 = pc.global.viewProj * modelTransform * f32v4(pos1, 1.0);
	const f32v4 ndc2 = pc.global.viewProj * modelTransform * f32v4(pos2, 1.0);

	const GpuDebugPC unf = GpuDebugPCBuffer(pc.global.debug).data;
	const ivec2 screensize = ivec2(unf.renderWidth, unf.renderHeight);
	const f32v2 clip0 = utilNdcToClip(ndc0, screensize);
	const f32v2 clip1 = utilNdcToClip(ndc1, screensize);
	const f32v2 clip2 = utilNdcToClip(ndc2, screensize);

	f32 barycentricDenom;
	const f32v3 barycentric = (
		utilBarycentric(
			f32v2(gl_GlobalInvocationID.xy) + 0.5f,
			clip0, clip1, clip2, barycentricDenom
		)
	);

	// -- compute perspective-correct barycentric coordinates
	const float invW0 = 1.0f / ndc0.w;
	const float invW1 = 1.0f / ndc1.w;
	const float invW2 = 1.0f / ndc2.w;
	const float perspectiveDenom = (
		  barycentric.x * invW0
		+ barycentric.y * invW1
		+ barycentric.z * invW2
	);

	// -- interpolate uv
	if (unpackUv) {
		retAttrData.uv = (
			(
				  attr0.uv * barycentric.x * invW0
				+ attr1.uv * barycentric.y * invW1
				+ attr2.uv * barycentric.z * invW2
			)
			/ perspectiveDenom
		);
	}

	// -- interpolate normal
	if (unpackNormal) {
		const f32v3 localNormal = (
			(
				  attr0.normal * barycentric.x * invW0
				+ attr1.normal * barycentric.y * invW1
				+ attr2.normal * barycentric.z * invW2
			)
			/ perspectiveDenom
		);
		const f32v4 localTangent = (
			(
				  attr0.tangent * barycentric.x * invW0
				+ attr1.tangent * barycentric.y * invW1
				+ attr2.tangent * barycentric.z * invW2
			)
			/ perspectiveDenom
		);
		// attributes are model-local; bring them into world space
		// (inverse-transpose handles non-uniform scale)
		const mat3 modelRot = mat3(modelTransform);
		const mat3 normalMat = transpose(inverse(modelRot));
		retAttrData.normal = normalize(normalMat * localNormal);
		retAttrData.tangent = f32v4(
			normalize(modelRot * localTangent.xyz),
			localTangent.w
		);
		retAttrData.normalGeometrical = (
			normalize(normalMat * cross(pos1 - pos0, pos2 - pos0))
		);
	}

	// -- interpolate world position and uv derivatives
	if (unpackWorldOrigin || unpackUvDerivatives) {
		const f32v3 objectPos = (
			(
				  pos0 * barycentric.x * invW0
				+ pos1 * barycentric.y * invW1
				+ pos2 * barycentric.z * invW2
			)
			/ perspectiveDenom
		);
		const f32v3 worldPos = (modelTransform * f32v4(objectPos, 1.0)).xyz;
		const f32v3 wPos0 = (modelTransform * f32v4(pos0, 1.0)).xyz;
		const f32v3 wPos1 = (modelTransform * f32v4(pos1, 1.0)).xyz;
		const f32v3 wPos2 = (modelTransform * f32v4(pos2, 1.0)).xyz;

		if (unpackWorldOrigin) {
			retAttrData.origin = worldPos;
		}

		if (unpackUvDerivatives) {
			// analytical barycentric derivatives
			const f32 duDx = (clip1.y - clip2.y) / barycentricDenom;
			const f32 dvDx = (clip2.y - clip0.y) / barycentricDenom;
			const f32 dwDx = (clip0.y - clip1.y) / barycentricDenom;
			const f32 duDy = (clip2.x - clip1.x) / barycentricDenom;
			const f32 dvDy = (clip0.x - clip2.x) / barycentricDenom;
			const f32 dwDy = (clip1.x - clip0.x) / barycentricDenom;

			const f32 perspDenomDx = duDx*invW0 + dvDx*invW1 + dwDx*invW2;
			const f32 perspDenomDy = duDy*invW0 + dvDy*invW1 + dwDy*invW2;
			const f32v2 uvNumerDx = (
				  attr0.uv*duDx*invW0
				+ attr1.uv*dvDx*invW1
				+ attr2.uv*dwDx*invW2
			);
			const f32v2 uvNumerDy = (
				  attr0.uv*duDy*invW0
				+ attr1.uv*dvDy*invW1
				+ attr2.uv*dwDy*invW2
			);
			retAttrData.uvDx = (
				(uvNumerDx - retAttrData.uv*perspDenomDx) / perspectiveDenom
			);
			retAttrData.uvDy = (
				(uvNumerDy - retAttrData.uv*perspDenomDy) / perspectiveDenom
			);
		}
	}

	// -- unpack material
	if (unpackMaterial) {
		retAttrData.material = (
			GpuMorMaterialBuffer(modelDesc.materials).data[meshlet.materialIndex]
		);
		retAttrData.materialIndex = meshlet.materialIndex;
		retAttrData.uvTransforms = GpuMorUvTransformBuffer(modelDesc.uvTransforms);
	}

	return retAttrData;
}

// -----------------------------------------------------------------------------
// -- model unpack from draw index and primitive index
// -----------------------------------------------------------------------------

struct UtilMeshAttributeDataFromIndices {
	GpuMorMaterial material;
	uint materialIndex;
	GpuMorUvTransformBuffer uvTransforms;
	f32v3 origin;
	f32v2 uv;
	f32v3 normal;
	f32v3 normalGeometrical;
	f32v4 tangent;
};

UtilMeshAttributeDataFromIndices utilMeshAttributeDataFromIndices(
	const uint modelDrawIndex,
	const uint primitiveIndex,
	const vec2 barycentric
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
	retAttrData.material = (
		GpuMorMaterialBuffer(model.materials).data[meshlet.materialIndex]
	);
	retAttrData.materialIndex = meshlet.materialIndex;
	retAttrData.uvTransforms = GpuMorUvTransformBuffer(model.uvTransforms);
	retAttrData.uv = (
		attr0.uv * (1.0 - barycentric.x - barycentric.y)
		+ attr1.uv * barycentric.x
		+ attr2.uv * barycentric.y
	);
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
	// tlas instance transform (the model matrix only, node transforms are not
	// baked into the blas), so bring them into world space to match the rays
	const mat3 modelRot = mat3(model.modelMatrix);
	const mat3 normalMat = transpose(inverse(modelRot));
	retAttrData.normal = normalize(normalMat * localNormal);
	retAttrData.normalGeometrical = (
		normalize(normalMat * cross(pos1 - pos0, pos2 - pos0))
	);
	retAttrData.tangent = f32v4(
		normalize(modelRot * localTangent.xyz),
		localTangent.w
	);
	return retAttrData;
}

#endif // UTIL_MESH_GLSL
