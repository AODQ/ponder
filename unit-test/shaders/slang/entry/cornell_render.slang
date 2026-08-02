import "util-types";

struct PC {
	u32v4 rootPad[8];
	u64 outVa;
	u32 width;
	u32 height;
};

[[vk::push_constant]] PC pc;

// brute-force lambertian path tracer over the open-front cornell box.
// primary rays are deterministic (no pixel jitter) so the host can classify
// light vs non-light pixels with the same camera math. output is three
// planar f32 channels (r, g, b) of the per-pixel radiance mean
static const f32v3 skCameraOrigin = f32v3(0.0, 0.0, -3.0);
static const f32 skTanHalfFov = 0.4;
static const u32 skSampleCount = 256u;
static const u32 skBounceCount = 8u;

static const f32 skTau = 6.28318530717958647692;

// wall order matches test-cornell.cpp: floor, ceiling, back, left (red),
// right (green), light
static const f32v3 skWallAlbedo[6] = {
	f32v3(0.73, 0.73, 0.73),
	f32v3(0.73, 0.73, 0.73),
	f32v3(0.73, 0.73, 0.73),
	f32v3(0.63, 0.065, 0.05),
	f32v3(0.14, 0.45, 0.091),
	f32v3(0.78, 0.78, 0.78)
};
static const f32v3 skWallEmission[6] = {
	f32v3(0.0), f32v3(0.0), f32v3(0.0), f32v3(0.0), f32v3(0.0),
	f32v3(15.0, 15.0, 15.0)
};
// plane normal per wall; sign is irrelevant, flipped to face the ray
static const f32v3 skWallNormals[6] = {
	f32v3(0.0, 1.0, 0.0),
	f32v3(0.0, -1.0, 0.0),
	f32v3(0.0, 0.0, -1.0),
	f32v3(1.0, 0.0, 0.0),
	f32v3(-1.0, 0.0, 0.0),
	f32v3(0.0, -1.0, 0.0)
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

// cosine-weighted hemisphere sample around n; pdf = cos(theta) / pi
func fnSampleCosineHemisphere(
	n: f32v3,
	u: f32v2,
) -> f32v3 {
	f32 r = sqrt(u.x);
	f32 phi = skTau * u.y;
	f32 lz = sqrt(max(0.0, 1.0 - u.x));
	// build an orthonormal basis around n (pixar / duff et al.)
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
[numthreads(8, 8, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	u32 px = gl_GlobalInvocationID.x;
	u32 py = gl_GlobalInvocationID.y;
	if (px >= pc.width || py >= pc.height) { return; }
	u32 pixel = py * pc.width + px;

	// ndc in [-1,1]; image row 0 is the top of the frame (+y in world)
	f32 ndcX = (2.0 * (f32(px) + 0.5) / f32(pc.width) - 1.0);
	f32 ndcY = (2.0 * (f32(py) + 0.5) / f32(pc.height) - 1.0);
	f32v3 primaryDir = normalize(
		f32v3(ndcX * skTanHalfFov, -ndcY * skTanHalfFov, 1.0)
	);

	u32 rngState = pixel * skSampleCount + 1u;
	f32v3 radianceSum = f32v3(0.0);

	for (u32 s = 0u; s < skSampleCount; ++s) {
		f32v3 origin = skCameraOrigin;
		f32v3 dir = primaryDir;
		f32v3 throughput = f32v3(1.0);

		for (u32 bounce = 0u; bounce < skBounceCount; ++bounce) {
			RayQuery<RAY_FLAG_NONE> rq;
			RayDesc rqRay;
			rqRay.Origin = origin;
			rqRay.TMin = // tMin is zero: self-intersection is prevented by the
				// origin offset along the normal; a positive tMin tunnels
				// through adjacent walls at corners
				0.0;
			rqRay.Direction = dir;
			rqRay.TMax = 100.0;
			rq.TraceRayInline(vkofTlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, rqRay);
			while (rq.Proceed()) { }

			bool hit = (
				rq.CommittedStatus()
				!= COMMITTED_NOTHING
			);
			// escaped through the open front face: no sky emission
			if (!hit) { break; }

			f32 t = rq.CommittedRayT();
			u32 wall = u32(rq.CommittedInstanceID());
			f32v3 hitPos = origin + t * dir;
			f32v3 n = skWallNormals[wall];
			if (dot(n, dir) > 0.0) { n = -n; }

			radianceSum += throughput * skWallEmission[wall];
			// lambertian, cosine-weighted sampling: f * cos / pdf = albedo
			throughput *= skWallAlbedo[wall];

			dir = fnSampleCosineHemisphere(n, fnSampleUniform2(rngState));
			origin = hitPos + n * 0.001;
		}
	}

	f32v3 radiance = radianceSum / f32(skSampleCount);
	u32 pixelCount = pc.width * pc.height;
	f32 * outBuf = (f32 *)(pc.outVa);
	outBuf[pixel] = radiance.r;
	outBuf[pixelCount + pixel] = radiance.g;
	outBuf[2u * pixelCount + pixel] = radiance.b;
}
