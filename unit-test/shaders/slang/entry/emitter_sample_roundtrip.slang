#include "../util-emitter-test.slang"


// the sampler-vs-pdf round trip: for every drawn sample, recompute the
// density from the sample's own (triangle, direction, distance) through
// utilEmitterPdf and report both. mis unbiasedness rests entirely on these
// two agreeing, and unlike the chi-square below a disagreement here is
// pinpointed to a single sample rather than smeared across a histogram.

// per sample: pdf from the sampler, pdf recomputed, distance, valid flag,
// primitiveIndex, spherical-branch flag, barycentric sum, non-finite flag
#define EMITTER_ROUNDTRIP_STRIDE 8u

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 sampleIndex = gl_GlobalInvocationID.x;
	if (sampleIndex >= pc.sampleCount) {
		return;
	}

	const u32 h0 = fnPcgHash(sampleIndex + fnPcgHash(pc.seedSalt));
	const u32 h1 = fnPcgHash(h0);
	const u32 h2 = fnPcgHash(h1);
	const u32 h3 = fnPcgHash(h2);
	const u32 h4 = fnPcgHash(h3);
	const f32v4 xi = f32v4(
		fnUintToUniform(h1), fnUintToUniform(h2),
		fnUintToUniform(h3), fnUintToUniform(h4)
	);

	GpuEmitterLight * lights = (GpuEmitterLight *)(pc.lightsVa);
	GpuResolveModelIndirect * models = (
		(GpuResolveModelIndirect *)(pc.modelsVa)
	);
	const EmitterSample s = (
		utilEmitterSample(lights, pc.lightCount, models, pc.from, xi)
	);

	f32 recomputed = 0.0f;
	f32 spherical = 0.0f;
	if (s.valid) {
		const GpuEmitterLight light = lights[s.lightIndex];
		const GpuResolveModelIndirect model = models[light.modelDrawIndex];
		recomputed = (
			utilEmitterPdf(
				lights, s.lightIndex, model, s.primitiveIndex,
				pc.from, s.direction, s.distance
			)
		);
		const EmitterTriangle tri = utilEmitterTriangle(model, s.primitiveIndex);
		const EmitterSphericalTriangle st = (
			utilEmitterSphericalTriangle(tri, pc.from)
		);
		spherical = utilEmitterUseSphericalSampling(st) ? 1.0f : 0.0f;
	}

	const bool nonFinite = (
		isnan(s.pdf) || isinf(s.pdf)
		|| isnan(recomputed) || isinf(recomputed)
		|| isnan(s.position.x) || isinf(s.position.x)
		|| isnan(s.position.y) || isinf(s.position.y)
		|| isnan(s.position.z) || isinf(s.position.z)
		|| isnan(s.direction.x) || isinf(s.direction.x)
		|| isnan(s.distance) || isinf(s.distance)
	);

	const u32 base = sampleIndex * EMITTER_ROUNDTRIP_STRIDE;
	((f32 *)(pc.outVa))[base + 0u] = s.pdf;
	((f32 *)(pc.outVa))[base + 1u] = recomputed;
	((f32 *)(pc.outVa))[base + 2u] = s.distance;
	((f32 *)(pc.outVa))[base + 3u] = s.valid ? 1.0f : 0.0f;
	((f32 *)(pc.outVa))[base + 4u] = f32(s.primitiveIndex);
	((f32 *)(pc.outVa))[base + 5u] = spherical;
	((f32 *)(pc.outVa))[base + 6u] = (
		s.barycentric.x + s.barycentric.y + s.barycentric.z
	);
	((f32 *)(pc.outVa))[base + 7u] = nonFinite ? 1.0f : 0.0f;
}
