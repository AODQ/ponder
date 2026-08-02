import "util-emitter";
import "util-mor-shared";
import "util-resolve-pc";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
};

[[vk::push_constant]] PC pc;

// util-emitter reads triangles out of GpuResolveModelIndirect; this file only
// exercises the geometry half, but the include has to resolve
#define RESOLVE_NO_PC

// the geometry half of util-emitter: triangle area/normal, the spherical
// triangle arvo 1995 samples in, and the solid-angle pdf. the pick and the
// hit lookup need a real emitter list and are covered separately

// in:  p0(3), p1(3), p2(3), from(3), xi(2)
// out: area, normal(3), valid, alpha, anglesSum, solidAngle, useSpherical,
//      bary(3), pointPdf, angleBetween, gramSchmidt(3)
[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 thread = gl_GlobalInvocationID.x;
	if (thread >= pc.threadCount) { return; }

	f32 * inBuf = (f32 *)(pc.inVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 i = thread * 14u;
	const u32 o = thread * 18u;

	EmitterTriangle tri;
	tri.p0 = f32v3(inBuf[i + 0u], inBuf[i + 1u], inBuf[i + 2u]);
	tri.p1 = f32v3(inBuf[i + 3u], inBuf[i + 4u], inBuf[i + 5u]);
	tri.p2 = f32v3(inBuf[i + 6u], inBuf[i + 7u], inBuf[i + 8u]);
	const f32v3 from = f32v3(
		inBuf[i + 9u], inBuf[i + 10u], inBuf[i + 11u]
	);
	const f32v2 xi = f32v2(inBuf[i + 12u], inBuf[i + 13u]);

	const EmitterTriangleGeometry geo = utilEmitterTriangleGeometry(tri);
	outBuf[o + 0u] = geo.area;
	outBuf[o + 1u] = geo.normal.x;
	outBuf[o + 2u] = geo.normal.y;
	outBuf[o + 3u] = geo.normal.z;

	const EmitterSphericalTriangle st = utilEmitterSphericalTriangle(tri, from);
	outBuf[o + 4u] = st.valid ? 1.0f : 0.0f;
	outBuf[o + 5u] = st.alpha;
	outBuf[o + 6u] = st.anglesSum;
	outBuf[o + 7u] = st.solidAngle;
	outBuf[o + 8u] = utilEmitterUseSphericalSampling(st) ? 1.0f : 0.0f;

	const f32v3 bary = (
		utilEmitterSampleSphericalTriangleBarycentric(tri, st, from, xi)
	);
	outBuf[o + 9u] = bary.x;
	outBuf[o + 10u] = bary.y;
	outBuf[o + 11u] = bary.z;

	// a point on the triangle from those barycentrics, so the pdf is asked
	// about a direction the sampler could actually have produced
	const f32v3 point = bary.x * tri.p0 + bary.y * tri.p1 + bary.z * tri.p2;
	const f32v3 toPoint = point - from;
	const f32 distance = length(toPoint);
	const f32v3 direction = distance > 0.0f ? toPoint / distance : f32v3(0.0f, 0.0f, 1.0f);
	outBuf[o + 12u] = utilEmitterPointPdf(tri, from, direction, distance);

	outBuf[o + 13u] = utilEmitterAngleBetween(geo.normal, direction);
	const f32v3 gs = utilEmitterGramSchmidt(tri.p1 - tri.p0, geo.normal);
	outBuf[o + 14u] = gs.x;
	outBuf[o + 15u] = gs.y;
	outBuf[o + 16u] = gs.z;
	outBuf[o + 17u] = distance;
}
