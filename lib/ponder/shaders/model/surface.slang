implementing model;

import types;

namespace vkof {

public enum GeometryKind : u32 {
	Triangle,
};

// a hit on a surface of any kind of geometry
public struct SurfaceHit {
	public GeometryKind kind;
	public u32 instance;
	// triangle: the primitive index
	public u32 element;
	// triangle: barycentrics
	public f32v2 parameters;
};

// vertex fetch + interpolation, independent of the material
public struct SurfaceGeometry {
	public f32v3 position;
	public f32v3 geometricNormal;
	public f32v3 shadingNormal;
	// w is the bitangent sign
	public f32v4 tangent;
	public f32v2 uv;
	// texture mip level
	public f32 textureLod;
	// world-space area of the primitive
	public f32 elementArea;
};

public func modelFromHit(hit: SurfaceHit) -> GpuResolveModelIndirect * {
	let models: GpuResolveModelIndirect * = (
		(GpuResolveModelIndirect *)(vkof::global.models)
	);
	return &models[hit.instance];
}

public func modelTriangleIndices(
	model: GpuResolveModelIndirect *,
	primitive: u32,
) -> u32v3 {
	let indices: u32 * = (u32 *)(model.flatIndices);
	return u32v3(
		indices[primitive * 3u + 0u],
		indices[primitive * 3u + 1u],
		indices[primitive * 3u + 2u]
	);
}

// interpolate a triangle attribute using barycentric coordinates
public func barycentricInterpolate<let N : int>(
	a: vector<f32, N>, b: vector<f32, N>, c: vector<f32, N>, barycentrics: f32v2
) -> vector<f32, N> {
	let w0: f32 = 1.0f - barycentrics.x - barycentrics.y;
	return a * w0 + b * barycentrics.x + c * barycentrics.y;
}

// transforms normal by the inverse-transpose of m
public func normalTransform(m: f32m3x3) -> f32m3x3 {
	let c0: f32v3 = f32v3(m[0][0], m[1][0], m[2][0]);
	let c1: f32v3 = f32v3(m[0][1], m[1][1], m[2][1]);
	let c2: f32v3 = f32v3(m[0][2], m[1][2], m[2][2]);
	let s: f32 = dot(c0, cross(c1, c2)) < 0.0f ? -1.0f : 1.0f;
	let n0: f32v3 = cross(c1, c2) * s;
	let n1: f32v3 = cross(c2, c0) * s;
	let n2: f32v3 = cross(c0, c1) * s;
	return f32m3x3(
		n0.x, n1.x, n2.x,
		n0.y, n1.y, n2.y,
		n0.z, n1.z, n2.z
	);
}

// uv only, for alpha testing inside traversal
public func modelHitUv(
	model: GpuResolveModelIndirect *,
	hit: SurfaceHit,
) -> f32v2 {
	let i: u32v3 = modelTriangleIndices(model, hit.element);
	let attributes: GpuMorVertexAttribute * = (
		(GpuMorVertexAttribute *)(model.attributes)
	);
	return barycentricInterpolate(
		attributes[i.x].uv, attributes[i.y].uv, attributes[i.z].uv,
		hit.parameters
	);
}

// creates a surface geometry just from a surface hit
public func surfaceGeometryFromParameters(
	hit: SurfaceHit,
) -> SurfaceGeometry {
	switch (hit.kind) {
		case GeometryKind.Triangle:
		default:
			return implSurfaceGeometryTriangle(hit);
	}
}

// creates a surface geometry from a surface hit and the traced world position
public func surfaceGeometryFromHit(
	hit: SurfaceHit,
	worldPosition: f32v3,
) -> SurfaceGeometry {
	var geometry: SurfaceGeometry = surfaceGeometryFromParameters(hit);
	geometry.position = worldPosition;
	return geometry;
}

} // namespace vkof

// -----------------------------------------------------------------------------
// -- private detail
// -----------------------------------------------------------------------------

func implSurfaceGeometryTriangle(
	hit: vkof::SurfaceHit,
) -> vkof::SurfaceGeometry {
	// -- fetch the model and triangle indices
	let model: GpuResolveModelIndirect * = vkof::modelFromHit(hit);
	let i: u32v3 = vkof::modelTriangleIndices(model, hit.element);
	let positions: f32v3 * = (f32v3 *)(model.positions);
	let attributes: GpuMorVertexAttribute * = (
		(GpuMorVertexAttribute *)(model.attributes)
	);

	// -- fetch the vertex attributes and positions
	let a0: GpuMorVertexAttribute = attributes[i.x];
	let a1: GpuMorVertexAttribute = attributes[i.y];
	let a2: GpuMorVertexAttribute = attributes[i.z];
	let p0: f32v3 = positions[i.x];
	let p1: f32v3 = positions[i.y];
	let p2: f32v3 = positions[i.z];

	// -- compute the world-space normal and tangent
	let modelRotation: f32m3x3 = f32m3x3(model.modelMatrix);
	let normalMatrix: f32m3x3 = vkof::normalTransform(modelRotation);
	let localNormal: f32v3 = (
		vkof::barycentricInterpolate(
			a0.normal, a1.normal, a2.normal, hit.parameters
		)
	);
	let localTangent: f32v4 = (
		vkof::barycentricInterpolate(
			a0.tangent, a1.tangent, a2.tangent, hit.parameters
		)
	);
	let worldNormal: f32v3 = mul(normalMatrix, localNormal);
	let worldTangent: f32v3 = mul(modelRotation, localTangent.xyz);

	// -- compute the world position from barycentric interpolation
	let localPosition: f32v3 = (
		vkof::barycentricInterpolate(p0, p1, p2, hit.parameters)
	);
	let worldPosition: f32v3 = (
		mul(model.modelMatrix, f32v4(localPosition, 1.0f)).xyz
	);

	// -- fill out the surface geometry
	var geometry: vkof::SurfaceGeometry;
	geometry.position = worldPosition;
	let faceNormal: f32v3 = mul(normalMatrix, cross(p1 - p0, p2 - p0));
	geometry.geometricNormal = normalize(faceNormal);
	// degenerate vertex normals fall back to the face normal
	geometry.shadingNormal = (
		dot(worldNormal, worldNormal) > 1e-12f
		? normalize(worldNormal)
		: geometry.geometricNormal
	);
	// -- compute tangent, uv
	geometry.tangent = (
		f32v4(
			f32v3(
				dot(worldTangent, worldTangent) > 1e-12f
				? normalize(worldTangent)
				: f32v3(0.0f)
			),
			localTangent.w
		)
	);
	geometry.uv = (
		vkof::barycentricInterpolate(a0.uv, a1.uv, a2.uv, hit.parameters)
	);
	geometry.textureLod = 0.0f;

	// compute the world-space area of the triangle
	geometry.elementArea = length(faceNormal) * 0.5f;

	return geometry;
}
