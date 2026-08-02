import "util-emitter";
import "util-mor-shared";
import "util-resolve-pc";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
	u64 modelsVa;
	u64 lightsVa;
};

[[vk::push_constant]] PC pc;

#define RESOLVE_NO_PC

// the half of util-emitter that reads a scene: triangle fetch through the
// model's flat index, the power-cdf pick, both pdfs, the full sample, and
// the backward hit lookup

// in:  primitiveIndex, lightCount, lightIndex, from(3), xi(4)
// out: tri.p0(3), area, pick(valid,index,pdf), pickPdf, pdf,
//      sample(valid,lightIndex,primitiveIndex,position(3),direction(3),
//      distance,bary(3),pdf), hit(found,lightIndex)
[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 thread = gl_GlobalInvocationID.x;
	if (thread >= pc.threadCount) { return; }

	f32 * inBuf = (f32 *)(pc.inVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 i = thread * 10u;
	const u32 o = thread * 24u;

	GpuResolveModelIndirect * models = (
		(GpuResolveModelIndirect *)(pc.modelsVa)
	);
	GpuEmitterLight * lights = (GpuEmitterLight *)(pc.lightsVa);
	const GpuResolveModelIndirect model = models[0];

	const u32 primitiveIndex = u32(inBuf[i + 0u]);
	const u32 lightCount = u32(inBuf[i + 1u]);
	const u32 lightIndex = u32(inBuf[i + 2u]);
	const f32v3 from = f32v3(
		inBuf[i + 3u], inBuf[i + 4u], inBuf[i + 5u]
	);
	const f32v4 xi4 = f32v4(
		inBuf[i + 6u], inBuf[i + 7u],
		inBuf[i + 8u], inBuf[i + 9u]
	);

	const EmitterTriangle tri = utilEmitterTriangle(model, primitiveIndex);
	outBuf[o + 0u] = tri.p0.x;
	outBuf[o + 1u] = tri.p0.y;
	outBuf[o + 2u] = tri.p0.z;
	outBuf[o + 3u] = utilEmitterTriangleGeometry(tri).area;

	const EmitterPick pick = utilEmitterPick(lights, lightCount, xi4.x);
	outBuf[o + 4u] = pick.valid ? 1.0f : 0.0f;
	outBuf[o + 5u] = f32(pick.lightIndex);
	outBuf[o + 6u] = pick.pickPdf;
	outBuf[o + 7u] = utilEmitterPickPdf(lights, lightIndex);

	const f32v3 toTri = tri.p0 - from;
	const f32 triDistance = length(toTri);
	const f32v3 triDirection = (
		triDistance > 0.0f ? toTri / triDistance : f32v3(0.0f, 0.0f, 1.0f)
	);
	outBuf[o + 8u] = utilEmitterPdf(
		lights, lightIndex, model, primitiveIndex, from, triDirection, triDistance
	);

	const EmitterSample smp = (
		utilEmitterSample(lights, lightCount, models, from, xi4)
	);
	outBuf[o + 9u] = smp.valid ? 1.0f : 0.0f;
	outBuf[o + 10u] = f32(smp.lightIndex);
	outBuf[o + 11u] = f32(smp.primitiveIndex);
	outBuf[o + 12u] = smp.position.x;
	outBuf[o + 13u] = smp.position.y;
	outBuf[o + 14u] = smp.position.z;
	outBuf[o + 15u] = smp.direction.x;
	outBuf[o + 16u] = smp.direction.y;
	outBuf[o + 17u] = smp.direction.z;
	outBuf[o + 18u] = smp.distance;
	outBuf[o + 19u] = smp.barycentric.x;
	outBuf[o + 20u] = smp.barycentric.y;
	outBuf[o + 21u] = smp.barycentric.z;

	const EmitterHitLookup hit = utilEmitterFindForHit(
		lights, model.emitterLightOffset, model.emitterLightCount, primitiveIndex
	);
	outBuf[o + 22u] = hit.found ? 1.0f : 0.0f;
	outBuf[o + 23u] = f32(hit.lightIndex);
}
