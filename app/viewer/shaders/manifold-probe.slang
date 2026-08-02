import "util-manifold-mesh";

// diagnostic manifold caustics,
// count out how often the solver finds bend point


struct ProbeCounts {
	// seeds tried, after the visibility reject
	u32 attempted;
	// seeds that reached a bend point
	u32 converged;
	// seeds abandoned because a reprojection ray left the caster
	u32 leftSurface;
	u32 pad;
};

// one per pixel. the glsl side reads this to light the receiver: where the
// bend point is, and whether one exists at all
struct BendPoint {
	f32v3 point;
	// 0 when no valid bend point was found for this pixel
	u32 converged;
	f32v3 normal;
	u32 pad;
};

struct ProbePush {
	// root constants occupy [0,128)
	u32v4 rootPad[8];
	u64 casterVa;
	f32v3 pointWo;
	u32 gbufferPosHandle;
	u32 gbufferNorHandle;
	u32 width;
	u32 height;
	u32 frameIndex;
	u64 countsVa;
	// BendPoint[width * height]
	u64 bendPointsVa;
};

[[vk::push_constant]] ProbePush pc;

u32 hash_u32(u32 v) {
	let state = v * 747796405u + 2891336453u;
	let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

f32 rand_f32(inout u32 state) {
	state = hash_u32(state);
	return f32(state >> 8u) * (1.0f / 16777216.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(const u32v3 tid : SV_DispatchThreadID) {
	if (tid.x >= pc.width || tid.y >= pc.height) { return; }

	// -- cleared first so the early-outs below leave no stale result behind
	BendPoint * bendPoints = (BendPoint *)pc.bendPointsVa;
	let pixel = tid.x + tid.y * pc.width;
	bendPoints[pixel] = BendPoint(f32v3(0.0f), 0u, f32v3(0.0f), 0u);

	// -- compute the bend point
	let coord = int2(tid.xy);
	let gbufferPos = vkofImages[pc.gbufferPosHandle][coord];
	// w <= 0 is a primary-ray miss, so there is no wi here
	if (gbufferPos.w <= 0.0f) { return; }

	let wiPoint = gbufferPos.xyz;
	let wiNormal = vkofImages[pc.gbufferNorHandle][coord].xyz;

	var rngState = (
		hash_u32(tid.x + tid.y * pc.width + pc.frameIndex * 9781u)
	);

	// -- seed a random point of the caster
	let caster = ((manifold.MeshData *)pc.casterVa)[0];
	let primitiveIndex = (
		u32(rand_f32(rngState) * f32(caster.triangleCount))
	);
	var u = rand_f32(rngState);
	var v = rand_f32(rngState);
	if (u + v > 1.0f) {
		u = 1.0f - u;
		v = 1.0f - v;
	}

	// -- compute guess
	let guessState = (
		manifold.mesh_point_state(
			caster,
			min(primitiveIndex, caster.triangleCount - 1u),
			f32v2(u, v),
			wiPoint
		)
	);

	// -- check if the guess is visible to the wo
	let seed = manifold.SurfaceMesh.sample(f32v2(0.0f), guessState);
	let toSeed = seed.point - wiPoint;
	if (dot(wiNormal, toSeed) <= 0.0f) { return; }
	if (dot(seed.normal, -normalize(toSeed)) <= 0.0f) { return; }

	// -- count the attempt
	Atomic<u32> * counts = (Atomic<u32> *)pc.countsVa;
	counts[0].add(1u);

	// -- solve for the bend point
	var path: manifold.Path;
	path.pointWi = wiPoint;
	path.pointWo = pc.pointWo;

	// entering glass from air
	path.eta = 1.5f;
	let result = manifold.solve<manifold.SurfaceMesh>(guessState, path);

	// -- count and publish the result
	if (result.converged) {
		counts[1].add(1u);
		let bend = manifold.SurfaceMesh.sample(f32v2(0.0f), result.bendPointState);
		bendPoints[pixel] = BendPoint(bend.point, 1u, bend.normal, 0u);
	} else if (result.iterCount == 0) {
		counts[2].add(1u);
	}
}
