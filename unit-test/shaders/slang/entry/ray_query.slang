import "util-types";

struct PC {
	u32v4 rootPad[8];
	u64 outVa;
};

[[vk::push_constant]] PC pc;

// Two invocations: 0 shoots toward the triangle, 1 shoots away.

[shader("compute")]
[numthreads(2, 1, 1)]
void main(
	const u32v3 gl_LocalInvocationID : SV_GroupThreadID,
) {
	// Triangle sits at z=1. Invocation 0 → hit, invocation 1 → miss.
	f32v3 dir = (
		(gl_LocalInvocationID.x == 0u)
		? f32v3(0.0, 0.0, 1.0)
		: f32v3(0.0, 0.0, -1.0)
	);

	RayQuery<RAY_FLAG_NONE> rq;
	RayDesc rqRay;
	rqRay.Origin = f32v3(0.0, 0.0, 0.0);
	rqRay.TMin = 0.001;
	rqRay.Direction = dir;
	rqRay.TMax = 100.0;
	rq.TraceRayInline(vkofTlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE, 0xFF, rqRay);
	rq.Proceed();

	bool hit = (
		rq.CommittedStatus()
		!= COMMITTED_NOTHING
	);
	((u32 *)(pc.outVa))[gl_LocalInvocationID.x] = hit ? 1u : 0u;
}
