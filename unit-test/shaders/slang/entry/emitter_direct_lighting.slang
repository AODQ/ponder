import "util-material-openpbr";
#include "../util-emitter-test.slang"
import "util-mis";


// tier 2 of plan-nee.md: the direct-lighting estimator itself, at a single
// shading vertex, computed three ways from the same rng stream -- light
// sampling alone, bsdf sampling alone, and the two MIS-combined. all three
// estimate the same integral, so they must agree; only their variance
// differs. running them together is what makes a wrong MIS weight visible: a
// dropped 1/p or a missing cosine moves all three equally, while a bad weight
// moves only the combined one.
//
// free space, so nothing occludes the emitter and no acceleration structure
// is needed -- the estimators reduce to exactly the terms under test.

// each invocation accumulates its own samples and writes three partial sums;
// the cpu reduces. float atomics would need an extension for no real gain
#define EMITTER_DIRECT_STRIDE 3u

func emitterTestMaterial(

) -> OpenPbrMaterial {
	OpenPbrMaterial mat;
	mat.baseWeight = pc.baseWeight;
	mat.baseColor = f32v3(1.0f);
	mat.baseMetalness = pc.metalness;
	mat.baseDiffuseRoughness = 0.0f;
	mat.specularWeight = pc.specularWeight;
	mat.specularColor = f32v3(1.0f);
	mat.specularRoughness = pc.specularRoughness;
	mat.specularRoughnessAnisotropy = 0.0f;
	mat.specularRoughnessAnisotropyRotation = 0.0f;
	mat.specularIor = 1.5f;
	mat.transmissionWeight = 0.0f;
	mat.transmissionColor = f32v3(1.0f);
	mat.transmissionDepth = 0.0f;
	mat.transmissionScatter = f32v3(0.0f);
	mat.transmissionScatterAnisotropy = 0.0f;
	mat.transmissionDispersionScale = 0.0f;
	mat.transmissionDispersionAbbeNumber = 20.0f;
	mat.subsurfaceWeight = 0.0f;
	mat.subsurfaceColor = f32v3(0.8f);
	mat.subsurfaceRadius = 1.0f;
	mat.subsurfaceRadiusScale = f32v3(1.0f);
	mat.subsurfaceScatterAnisotropy = 0.0f;
	mat.coatWeight = 0.0f;
	mat.coatColor = f32v3(1.0f);
	mat.coatRoughness = 0.0f;
	mat.coatRoughnessAnisotropy = 0.0f;
	mat.coatIor = 1.6f;
	mat.coatDarkening = 1.0f;
	mat.fuzzWeight = 0.0f;
	mat.fuzzColor = f32v3(1.0f);
	mat.fuzzRoughness = 0.5f;
	mat.emissionLuminance = 0.0f;
	mat.emissionColor = f32v3(0.0f);
	mat.thinFilmWeight = 0.0f;
	mat.thinFilmThickness = 0.5f;
	mat.thinFilmIor = 1.4f;
	mat.geometryOpacity = 1.0f;
	mat.geometryThinWalled = 0.0f;
	return mat;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 threadIndex = gl_GlobalInvocationID.x;
	if (threadIndex >= pc.threadCount) {
		return;
	}

	MaterialTableHandles tables;
	tables.kullaContyEnergy = pc.kullaContyHandle;
	tables.zeltnerLtcParam = 0u;

	const OpenPbrMaterial mat = emitterTestMaterial();
	const f32v3 nor = normalize(pc.shadingNormal);
	const ShadingFrame frame = shadingFrameFromNormal(nor);
	const ShadingFrame coatFrame = frame;
	// a fixed oblique incidence rather than straight on, so a cosine dropped
	// from one estimator cannot cancel against the geometry
	const f32v3 wi = normalize(f32v3(0.35f, 0.85f, 0.2f));

	GpuEmitterLight * lights = (GpuEmitterLight *)(pc.lightsVa);
	GpuResolveModelIndirect * models = (
		(GpuResolveModelIndirect *)(pc.modelsVa)
	);

	f32v3 sumNee = f32v3(0.0f);
	f32v3 sumBsdf = f32v3(0.0f);
	f32v3 sumMis = f32v3(0.0f);

	const u32 h0 = fnPcgHash(threadIndex + fnPcgHash(pc.seedSalt));
	const u32 h1 = fnPcgHash(h0);
	u64 state = (u64(h0) << 32) | u64(h1);

	for (u32 s = 0u; s < pc.samplesPerThread; ++s) {
		// -- light sampling
		{
			const f32v4 xi = f32v4(
				fnSampleUniform(state), fnSampleUniform(state),
				fnSampleUniform(state), fnSampleUniform(state)
			);
			const EmitterSample es = (
				utilEmitterSample(lights, pc.lightCount, models, pc.from, xi)
			);
			if (es.valid) {
				const f32 dotNorWo = dot(nor, es.direction);
				if (dotNorWo > 0.0f) {
					const f32v3 f = (
						openPbrEvaluateF(
							tables, frame, coatFrame, wi, es.direction, mat,
							false, 1.5f
						)
					);
					const f32 bsdfPdf = (
						openPbrEvaluatePdf(
							tables, mat, frame, coatFrame, wi, es.direction,
							false, 1.5f
						)
					);
					const GpuEmitterLight light = lights[es.lightIndex];
					const f32v3 contribution = (
						light.radiance * f * dotNorWo / es.pdf
					);
					sumNee += contribution;
					sumMis += contribution * misBalanceWeight(es.pdf, bsdfPdf);
				}
			}
		}

		// -- bsdf sampling
		{
			f32 bsdfPdf;
			bool sampledTransmission;
			bool sampledSubsurfaceEntry;
			u32 pickedLobe;
			const f32v3 wo = (
				openPbrSampleWo(
					tables, frame, coatFrame, wi, mat, bsdfPdf,
					sampledTransmission, sampledSubsurfaceEntry, pickedLobe,
					state, false, 1.5f
				)
			);
			const f32 dotNorWo = dot(nor, wo);
			if (bsdfPdf > 0.0f && dotNorWo > 0.0f) {
				u32 lightIndex;
				u32 primitiveIndex;
				f32 hitDistance;
				if (
					emitterTestNearestHit(
						pc.from, wo, lightIndex, primitiveIndex, hitDistance
					)
				) {
					const GpuEmitterLight light = lights[lightIndex];
					const GpuResolveModelIndirect model = (
						models[light.modelDrawIndex]
					);
					const f32v3 f = (
						openPbrEvaluateF(
							tables, frame, coatFrame, wi, wo, mat, false, 1.5f
						)
					);
					const f32 lightPdf = (
						utilEmitterPdf(
							lights, lightIndex, model, primitiveIndex, pc.from,
							wo, hitDistance
						)
					);
					const f32v3 contribution = (
						light.radiance * f * dotNorWo / bsdfPdf
					);
					sumBsdf += contribution;
					sumMis += contribution * misBalanceWeight(bsdfPdf, lightPdf);
				}
			}
		}
	}

	const f32 invSamples = 1.0f / f32(pc.samplesPerThread);
	const u32 base = threadIndex * EMITTER_DIRECT_STRIDE;
	((f32 *)(pc.outVa))[base + 0u] = (
		dot(sumNee, f32v3(1.0f / 3.0f)) * invSamples
	);
	((f32 *)(pc.outVa))[base + 1u] = (
		dot(sumBsdf, f32v3(1.0f / 3.0f)) * invSamples
	);
	((f32 *)(pc.outVa))[base + 2u] = (
		dot(sumMis, f32v3(1.0f / 3.0f)) * invSamples
	);
}
