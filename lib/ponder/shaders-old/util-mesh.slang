module "util-mesh";
__exported import "util-types";
__exported import "util-mor-shared";
__exported import "util-resolve-pc";
// util to pack and unpack mesh data


// -----------------------------------------------------------------------------
// -- 32-bit packing of mesh data
// -----------------------------------------------------------------------------

public struct UtilMeshUnpacked {
	public u32 triangleId;
	public u32 meshletId;
	public u32 modelId;
};

public func utilMeshUnpackTexel(
	packed: u32,
) -> UtilMeshUnpacked {
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

// carries the material buffer reference + index
public struct UtilMeshAttributeData {
	public GpuMorMaterial * materialBuf;
	public u32 materialIndex;
	public GpuMorUvTransform * uvTransforms;
	public f32v3 origin;
	public f32v2 uv;
	public f32v2 uvDx;
	public f32v2 uvDy;
	public f32v3 normal;
	public f32v4 tangent;
	public f32v3 normalGeometrical;
};

public func utilBarycentric(
	p: f32v2,
	a: f32v2,
	b: f32v2,
	c: f32v2,
	out outDenom: f32,
) -> f32v3 {
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
	return f32v3(u, v, 1.0f - u - v);
}

// ndc to clip is in vulkan conventions
public func utilNdcToClip(
	ndc: f32v4,
	screensize: i32v2,
) -> f32v2 {
	const f32v2 screen = ndc.xy / ndc.w * 0.5f + 0.5f;
	return screen * f32v2(screensize);
}

// what utilMeshDecodeAttributes needs from the frame's globals. passed
// explicitly rather than read from pc: a slang module cannot see the entry
// point's push constant
public struct UtilMeshDecodeContext {
	public u64 models;
	public f32m44 viewProj;
	// the pixel this invocation resolves; slang reaches the invocation id
	// only as an entry-point semantic, never as a free global
	public u32v2 invocationId;
};

public func utilMeshDecodeAttributes(
	ctx: UtilMeshDecodeContext,
	texelUnpacked: UtilMeshUnpacked,
	screensize: i32v2,
	unpackWorldOrigin: bool,
	unpackNormal: bool,
	unpackUv: bool,
	unpackUvDerivatives: bool,
	unpackMaterial: bool,
) -> UtilMeshAttributeData {
	const u32 modelId = texelUnpacked.modelId;
	const u32 meshletId = texelUnpacked.meshletId;
	const u32 triangleId = texelUnpacked.triangleId;

	UtilMeshAttributeData retAttrData;

	// -- unpack model/meshlet/triangle data
	GpuResolveModelIndirect * modelsBuf = (
		(GpuResolveModelIndirect *)(ctx.models)
	);
	const GpuResolveModelIndirect modelDesc = modelsBuf[modelId];
	const GpuMorMeshlet meshlet = (
		((GpuMorMeshlet *)(modelDesc.meshlets))[meshletId]
	);
	uint8_t * meshletTriBuffer = (
		(uint8_t *)(modelDesc.meshletTris)
	);
	u32 * meshletVertBuffer = (
		(u32 *)(modelDesc.meshletVerts)
	);
	const GpuMorInstance instance = (
		((GpuMorInstance *)(modelDesc.instances))[meshlet.instanceIndex]
	);

	// -- unpack triangle vertex indices
	const u32 triangleBaseIdx = meshlet.triangleOffset + triangleId * 3;
	const u32 localIdx0 = u32(meshletTriBuffer[triangleBaseIdx + 0]);
	const u32 localIdx1 = u32(meshletTriBuffer[triangleBaseIdx + 1]);
	const u32 localIdx2 = u32(meshletTriBuffer[triangleBaseIdx + 2]);

	// -- convert local vertex indices to global vertex indices
	const u32 globalIdx0 = (
		meshletVertBuffer[meshlet.vertexOffset + localIdx0]
	);
	const u32 globalIdx1 = (
		meshletVertBuffer[meshlet.vertexOffset + localIdx1]
	);
	const u32 globalIdx2 = (
		meshletVertBuffer[meshlet.vertexOffset + localIdx2]
	);

	// -- unpack vertex positions
	f32v3 * positionBuf = (
		(f32v3 *)(modelDesc.positions)
	);
	const f32v3 pos0 = positionBuf[globalIdx0];
	const f32v3 pos1 = positionBuf[globalIdx1];
	const f32v3 pos2 = positionBuf[globalIdx2];

	// -- unpack attributes
	GpuMorVertexAttribute * attrBuf = (
		(GpuMorVertexAttribute *)(modelDesc.attributes)
	);
	const GpuMorVertexAttribute attr0 = attrBuf[globalIdx0];
	const GpuMorVertexAttribute attr1 = attrBuf[globalIdx1];
	const GpuMorVertexAttribute attr2 = attrBuf[globalIdx2];

	// -- reconstruct clip space
	const f32m44 modelTransform = (
		mul(modelDesc.modelMatrix, instance.transform));
	const f32v4 ndc0 = mul(ctx.viewProj, mul(modelTransform, f32v4(pos0, 1.0)));
	const f32v4 ndc1 = mul(ctx.viewProj, mul(modelTransform, f32v4(pos1, 1.0)));
	const f32v4 ndc2 = mul(ctx.viewProj, mul(modelTransform, f32v4(pos2, 1.0)));

	const f32v2 clip0 = utilNdcToClip(ndc0, screensize);
	const f32v2 clip1 = utilNdcToClip(ndc1, screensize);
	const f32v2 clip2 = utilNdcToClip(ndc2, screensize);

	f32 barycentricDenom;
	const f32v3 barycentric = (
		utilBarycentric(
			f32v2(ctx.invocationId) + 0.5f,
			clip0, clip1, clip2, barycentricDenom
		)
	);

	// -- compute perspective-correct barycentric coordinates
	const f32 invW0 = 1.0f / ndc0.w;
	const f32 invW1 = 1.0f / ndc1.w;
	const f32 invW2 = 1.0f / ndc2.w;
	const f32 perspectiveDenom = (
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
		const f32m3x3 modelRot = f32m3x3(modelTransform);
		const f32m3x3 normalMat = transpose(inverse(modelRot));
		retAttrData.normal = normalize(mul(normalMat, localNormal));
		retAttrData.tangent = f32v4(
			normalize(mul(modelRot, localTangent.xyz)),
			localTangent.w
		);
		retAttrData.normalGeometrical = (
			normalize(mul(normalMat, cross(pos1 - pos0, pos2 - pos0)))
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
		const f32v3 worldPos = (mul(modelTransform, f32v4(objectPos, 1.0))).xyz;

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
		retAttrData.materialBuf = (GpuMorMaterial *)(modelDesc.materials);
		retAttrData.materialIndex = meshlet.materialIndex;
		retAttrData.uvTransforms = (GpuMorUvTransform *)(modelDesc.uvTransforms);
	}

	return retAttrData;
}

