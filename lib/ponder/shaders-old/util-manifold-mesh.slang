module "util-manifold-mesh";
__exported import "util-manifold";

// manifold surface over triangle geometry.
//
// the analytic surfaces extend forever, but during the newton solve the
// bend-point state could leave the triangle it starts on. Thus the solver is
// recalculating the triangle's vertices and normals based off the mesh surface.
// this requires ray queries into an acceleration structure


public namespace manifold
{

// mirrors GpuMorVertexAttribute
public struct MeshVertexAttribute {
	public f32v3 normal;
	public f32v2 uv;
	public f32v4 tangent;
};

// mor indexed triangle geometry
public struct MeshData {
	public u64 positionVa;
	public u64 attributeVa;
	public u64 indexVa;
	public f32m4x4 modelMatrix;
	public u32 triangleCount;
	public u32 pad;
};

// bend-point state for a triangle mesh.
public struct MeshPointState {
	public MeshData mesh;
	public u32 primitiveIndex;
	public f32v2 uv;

	// for performing ray queries to find the next triangle when moving the
	// bend-point state
	public f32v3 rayOrigin;

	public f32v3 p0;
	public f32v3 p1;
	public f32v3 p2;
	public f32v3 nor0;
	public f32v3 nor1;
	public f32v3 nor2;
}

// inverse transpose of a 3x3
// slang has no matrix inverse
func mat3_inverse_transpose(m: f32m3x3) -> f32m3x3 {
	let cofactor = f32m3x3(
		m[1][1] * m[2][2] - m[1][2] * m[2][1],
		m[1][2] * m[2][0] - m[1][0] * m[2][2],
		m[1][0] * m[2][1] - m[1][1] * m[2][0],

		m[0][2] * m[2][1] - m[0][1] * m[2][2],
		m[0][0] * m[2][2] - m[0][2] * m[2][0],
		m[0][1] * m[2][0] - m[0][0] * m[2][1],

		m[0][1] * m[1][2] - m[0][2] * m[1][1],
		m[0][2] * m[1][0] - m[0][0] * m[1][2],
		m[0][0] * m[1][1] - m[0][1] * m[1][0]
	);
	let det = (
		m[0][0] * cofactor[0][0]
		+ m[0][1] * cofactor[0][1]
		+ m[0][2] * cofactor[0][2]
	);
	return cofactor / det;
}

// loads primitiveIndex's vertices into the cache, in world space
public func mesh_point_state_fetch(inout bendPointState: MeshPointState) {
	let mesh = bendPointState.mesh;
	f32v3 * positions = (f32v3 *)mesh.positionVa;
	MeshVertexAttribute * attributes = (MeshVertexAttribute *)mesh.attributeVa;
	u32 * indices = (u32 *)mesh.indexVa;

	let i0 = indices[bendPointState.primitiveIndex * 3u + 0u];
	let i1 = indices[bendPointState.primitiveIndex * 3u + 1u];
	let i2 = indices[bendPointState.primitiveIndex * 3u + 2u];

	let modelRot = f32m3x3(mesh.modelMatrix);
	let normalMatrix = mat3_inverse_transpose(modelRot);

	bendPointState.p0 = mul(mesh.modelMatrix, f32v4(positions[i0], 1.0f)).xyz;
	bendPointState.p1 = mul(mesh.modelMatrix, f32v4(positions[i1], 1.0f)).xyz;
	bendPointState.p2 = mul(mesh.modelMatrix, f32v4(positions[i2], 1.0f)).xyz;
	bendPointState.nor0 = normalize(mul(normalMatrix, attributes[i0].normal));
	bendPointState.nor1 = normalize(mul(normalMatrix, attributes[i1].normal));
	bendPointState.nor2 = normalize(mul(normalMatrix, attributes[i2].normal));
}

// bend-point state at a triangle's barycentrics.
public func mesh_point_state(
	mesh: MeshData,
	primitiveIndex: u32,
	uv: f32v2,
	rayOrigin: f32v3,
) -> MeshPointState {
	var bendPointState: MeshPointState;
	bendPointState.mesh = mesh;
	bendPointState.primitiveIndex = primitiveIndex;
	bendPointState.uv = uv;
	bendPointState.rayOrigin = rayOrigin;
	mesh_point_state_fetch(bendPointState);
	return bendPointState;
}

// surface of a triangle mesh
public struct SurfaceMesh : ISurface {
	public typealias PointState = MeshPointState;

	[Differentiable]
	public static func sample(
		x: f32v2,
		no_diff bendPointState: MeshPointState,
	) -> Sample {
		let uv = bendPointState.uv + x;
		let w = 1.0f - uv.x - uv.y;
		let point = (
			w * bendPointState.p0
			+ uv.x * bendPointState.p1
			+ uv.y * bendPointState.p2
		);
		let nor = normalize(
			w * bendPointState.nor0
			+ uv.x * bendPointState.nor1
			+ uv.y * bendPointState.nor2
		);
		let edge = bendPointState.p1 - bendPointState.p0;
		return Sample(point, nor, frame_from_normal(nor, edge));
	}

	// will return false if the ray query fails to hit a triangle when
	// moving the bend-point state to a new triangle
	public static func advance(
		inout bendPointState: MeshPointState,
		x: f32v2,
	) -> bool {
		let target = sample(x, bendPointState).point;

		RayDesc ray;
		ray.Origin = bendPointState.rayOrigin;
		ray.TMin = 1e-4;
		ray.Direction = normalize(target - bendPointState.rayOrigin);
		ray.TMax = 1e4;

		RayQuery<RAY_FLAG_CULL_NON_OPAQUE> rq;
		rq.TraceRayInline(vkofTlas, RAY_FLAG_NONE, 0xFF, ray);
		rq.Proceed();
		if (rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT) { return false; }

		bendPointState.primitiveIndex = rq.CommittedPrimitiveIndex();
		bendPointState.uv = rq.CommittedTriangleBarycentrics();
		mesh_point_state_fetch(bendPointState);
		return true;
	}
}

} // namespace manifold
