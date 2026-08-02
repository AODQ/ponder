// forward-mode autodiff against central differences, on the shape
// plan-slang-caustics.md actually needs: the half-vector specular
// constraint as a function of a bend point's barycentric coordinates.
//
// each thread solves one configuration and writes both jacobians; the cpu
// compares them

struct TriangleData {
	float3 p0;
	float3 p1;
	float3 p2;
	float3 n0;
	float3 n1;
	float3 n2;
};

struct Config {
	TriangleData tri;
	float3 receiver;
	float3 light;
	float2 uv;
	float eta;
	float pad;
};

struct PushConstants {
	uint4 rootPad[8];
	uint64_t configVa;
	uint64_t outVa;
	uint configCount;
	float fdStep;
};

[[vk::push_constant]] PushConstants pc;

// zero when the path through uv obeys snell's law at the bend point
[Differentiable]
float2 half_vector_constraint(
	float2 uv,
	no_diff TriangleData tri,
	no_diff float3 receiver,
	no_diff float3 light,
	no_diff float eta
) {
	const float3 p = (
		tri.p0 + uv.x * (tri.p1 - tri.p0) + uv.y * (tri.p2 - tri.p0)
	);
	const float3 n = normalize(
		tri.n0 + uv.x * (tri.n1 - tri.n0) + uv.y * (tri.n2 - tri.n0)
	);
	const float3 wi = normalize(receiver - p);
	const float3 wo = normalize(light - p);
	const float3 h = normalize(wi + eta * wo);

	// tangent frame seeded off the triangle edge, so the frame itself
	// varies with uv exactly as the reference implementation's does
	const float3 edge = tri.p1 - tri.p0;
	const float3 s = normalize(edge - n * dot(edge, n));
	const float3 t = cross(n, s);
	return float2(dot(h, s), dot(h, t));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	if (tid.x >= pc.configCount) { return; }

	Config * configs = (Config *)pc.configVa;
	const Config c = configs[tid.x];
	float * out = (float *)pc.outVa;

	// -- autodiff: one forward pass per input seeds one jacobian column
	const DifferentialPair<float2> du = fwd_diff(half_vector_constraint)(
		diffPair(c.uv, float2(1.0, 0.0)),
		c.tri, c.receiver, c.light, c.eta
	);
	const DifferentialPair<float2> dv = fwd_diff(half_vector_constraint)(
		diffPair(c.uv, float2(0.0, 1.0)),
		c.tri, c.receiver, c.light, c.eta
	);

	// -- central differences
	const float h = pc.fdStep;
	const float2 uPlus = half_vector_constraint(
		c.uv + float2(h, 0.0), c.tri, c.receiver, c.light, c.eta
	);
	const float2 uMinus = half_vector_constraint(
		c.uv - float2(h, 0.0), c.tri, c.receiver, c.light, c.eta
	);
	const float2 vPlus = half_vector_constraint(
		c.uv + float2(0.0, h), c.tri, c.receiver, c.light, c.eta
	);
	const float2 vMinus = half_vector_constraint(
		c.uv - float2(0.0, h), c.tri, c.receiver, c.light, c.eta
	);

	const uint base = tid.x * 10u;
	out[base + 0u] = du.d.x;
	out[base + 1u] = du.d.y;
	out[base + 2u] = dv.d.x;
	out[base + 3u] = dv.d.y;
	out[base + 4u] = (uPlus.x - uMinus.x) / (2.0 * h);
	out[base + 5u] = (uPlus.y - uMinus.y) / (2.0 * h);
	out[base + 6u] = (vPlus.x - vMinus.x) / (2.0 * h);
	out[base + 7u] = (vPlus.y - vMinus.y) / (2.0 * h);
	// the primal, so a config misread as the wrong struct layout fails here
	// rather than passing with both jacobians equally wrong
	out[base + 8u] = du.p.x;
	out[base + 9u] = du.p.y;
}
