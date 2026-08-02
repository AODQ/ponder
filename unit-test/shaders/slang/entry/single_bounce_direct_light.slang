import "util-types";

struct PC {
	u32v4 rootPad[8];
	u64 shadeVa;
	u64 outVa;
	u32 shadeCount;
	u32 sampleCount;
	f32 albedo;
	f32 emission;
};

[[vk::push_constant]] PC pc;

// layer 1 of the path-tracer verification effort: single-bounce direct
// lighting. a primary ray hits a lambertian plane (instance 0); exactly
// one more bounce is sampled (cosine-weighted brdf importance sampling,
// no next-event-estimation -- none exists in this harness yet). if that
// bounce hits the light quad (instance 1) the sample contributes
// albedo * emission (the usual f*cos/pdf = albedo cancellation for a
// cosine-sampled lambertian lobe), otherwise zero. ties real bvh geometry
// (layer 0) to a real shading bounce, isolated from any further/indirect
// bounces so the estimator's mean has a closed-form (well, quadrature)
// ground truth on the cpu side.
struct ShadePoint {
	f32v3 origin;
	f32v3 direction;
};

func fnPcgHash(
	v: u32,
) -> u32 {
	u32 state = v * 747796405u + 2891336453u;
	u32 word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

func fnSampleUniform2(
	inout state: u32,
) -> f32v2 {
	state = fnPcgHash(state);
	u32 a = state;
	state = fnPcgHash(state);
	u32 b = state;
	return f32v2(
		asfloat((a >> 9u) | 0x3F800000u) - 1.0,
		asfloat((b >> 9u) | 0x3F800000u) - 1.0
	);
}

static const f32 skTau = 6.28318530717958647692;

// cosine-weighted hemisphere sample around n; pdf = cos(theta) / pi
func fnSampleCosineHemisphere(
	n: f32v3,
	u: f32v2,
) -> f32v3 {
	f32 r = sqrt(u.x);
	f32 phi = skTau * u.y;
	f32 lz = sqrt(max(0.0, 1.0 - u.x));
	f32 s = (n.z >= 0.0) ? 1.0 : -1.0;
	f32 a = -1.0 / (s + n.z);
	f32 b = n.x * n.y * a;
	f32v3 t = f32v3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
	f32v3 bt = f32v3(b, s + n.y * n.y * a, -n.y);
	return normalize(
		t * (r * cos(phi)) + bt * (r * sin(phi)) + n * lz
	);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 idx = gl_GlobalInvocationID.x;
	if (idx >= pc.shadeCount) { return; }

	ShadePoint * shadePts = (ShadePoint *)(pc.shadeVa);
	ShadePoint sp = shadePts[idx];

	RayQuery<RAY_FLAG_NONE> primary;
	RayDesc primaryRay;
	primaryRay.Origin = sp.origin;
	primaryRay.TMin = 0.0;
	primaryRay.Direction = sp.direction;
	primaryRay.TMax = 100.0;
	primary.TraceRayInline(vkofTlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, primaryRay);
	while (primary.Proceed()) { }
	bool primaryHit = (
		primary.CommittedStatus()
		!= COMMITTED_NOTHING
	);

	f32 sum = 0.0;
	if (primaryHit) {
		f32 t = primary.CommittedRayT();
		f32v3 p = sp.origin + t * sp.direction;
		f32v3 n = f32v3(0.0, 1.0, 0.0);
		u32 state = idx * 9781u + 1u;

		for (u32 s = 0u; s < pc.sampleCount; ++s) {
			f32v3 wo = fnSampleCosineHemisphere(n, fnSampleUniform2(state));
			RayQuery<RAY_FLAG_NONE> rq;
			RayDesc rqRay;
			rqRay.Origin = p + n * 0.001;
			rqRay.TMin = 0.0;
			rqRay.Direction = wo;
			rqRay.TMax = 100.0;
			rq.TraceRayInline(vkofTlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, rqRay);
			while (rq.Proceed()) { }
			bool hit = (
				rq.CommittedStatus()
				!= COMMITTED_NOTHING
			);
			if (hit) {
				u32 inst = u32(
					rq.CommittedInstanceID()
				);
				if (inst == 1u) {
					sum += pc.albedo * pc.emission;
				}
			}
		}
	}
	((f32 *)(pc.outVa))[idx] = sum / f32(pc.sampleCount);
}
