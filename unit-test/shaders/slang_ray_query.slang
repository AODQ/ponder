// slang counterpart of ray_query.comp: same triangle, same two rays, same
// tlas binding. proves slang reaches the acceleration structure the manifold
// solver's reprojection step needs

struct PushConstants {
	uint4 rootPad[8];
	uint64_t outVa;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(3, 0)]] RaytracingAccelerationStructure vkofTlas;

[shader("compute")]
[numthreads(2, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	// triangle sits at z=1; invocation 0 shoots toward it, 1 away
	const float3 dir = (
		(tid.x == 0u) ? float3(0.0, 0.0, 1.0) : float3(0.0, 0.0, -1.0)
	);

	RayDesc ray;
	ray.Origin = float3(0.0, 0.0, 0.0);
	ray.TMin = 0.001;
	ray.Direction = dir;
	ray.TMax = 100.0;

	RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_NON_OPAQUE> rq;
	rq.TraceRayInline(vkofTlas, RAY_FLAG_NONE, 0xFF, ray);
	rq.Proceed();

	const bool hit = (
		rq.CommittedStatus() != COMMITTED_NOTHING
	);
	uint * out = (uint *)pc.outVa;
	out[tid.x] = hit ? 1u : 0u;
}
