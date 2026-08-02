import "util-material-openpbr";
import "util-environment-map";
import "util-mis";
import "util-env-dl-material";

struct PC {
	u32v4 rootPad[8];
	u64 sumVa;
	u64 sumSqVa;
	u64 badVa;
	EnvironmentMapHandles handles;
	u32 kullaContyEnergyHandle;
	u32 zeltnerLtcParamHandle;
	u32 mode;
	u32 threadCount;
	u32 samplesPerThread;
	u32 seedSalt;
	f32 rotation;
	f32v3 wi;
	f32v3 baseColor;
	f32 baseMetalness;
	f32 specularRoughness;
	f32 specularIor;
	f32 coatWeight;
	f32 coatRoughness;
	f32 coatIor;
	// > 0 applies resolve.comp's fnFireflyClamp to each sample before it is
	// accumulated. a well-importance-sampled estimator should barely notice
	// a clamp set above its typical value; one that is quietly producing
	// mispriced outliers loses real energy to it
	f32 fireflyClampLuminance;
	// diagnostic: read radiance with texelFetch instead of the production
	// bilinear texture(). the pdf table is exact per texel, so a bilinear
	// radiance means a texel adjacent to a sun still returns part of that
	// sun while being priced at its own (sky) density. switching the fetch
	// to nearest removes exactly that mechanism and nothing else, which is
	// what attributes the clamp loss below to it or to the sub-texel
	// mispricing. NOT what the renderer does -- comparison only
	u32 radianceNearest;
};

[[vk::push_constant]] PC pc;

// direct-lighting estimator for a single surface vertex under an hdr
// environment map, run three ways over the same material and the same map:
//
//   mode 0 (nee)  -- env importance sampling alone, no mis weight
//   mode 1 (bsdf) -- openPbrSampleWo alone, no mis weight
//   mode 2 (mis)  -- both halves with balance weights, byte-for-byte the
//                    estimator app/viewer/shaders/util-raytrace.glsl runs
//                    (the nee block at :319 plus the env-escape reweight
//                    at :1515)
//
// all three estimate the same integral,
//
//   \int_{S^2} L(\omega) f(\omega_i, \omega) |\cos\theta| d\omega
//
// so their means must coincide with each other and with the deterministic
// quadrature in env_direct_lighting_reference.comp. no geometry and no
// shadow rays: this isolates the sampling/pdf/mis arithmetic from
// self-intersection and scene scale, so a disagreement here can only come
// from the estimator itself.
//
// the mis weights are the load-bearing part. where the map is locally very
// bright, envPdf >> bsdfPdf, so weightB -> 0 and essentially the whole
// contribution has to arrive through the nee half; anything that makes the
// nee half under-deliver shows up as energy loss proportional to envPdf --
// i.e. only in the bright parts of the map.

#define ENV_DL_MODE_NEE 0u
#define ENV_DL_MODE_BSDF 1u
#define ENV_DL_MODE_MIS 2u

// sumVa: threadCount * 4 floats, this thread's running sum -- rgb, then the
// per-sample luminance
// sumSqVa: threadCount * 4 floats, the matching sums of squares, so the
// caller can turn each estimator's own sample variance into a standard error
// and test the mean against it instead of against a guessed tolerance
// badVa: slot 0 counts non-finite per-sample estimator values

func fnEnvRadiance(
	dir: f32v3,
) -> f32v3 {
	if (pc.radianceNearest == 0u) {
		return environmentMapRadiance(pc.handles, dir, pc.rotation);
	}
	const i32v2 size = textureSize(vkofTextures[pc.handles.radiance], 0);
	const f32v2 uv = environmentMapDirToUv(dir, pc.rotation);
	const i32v2 texel = i32v2(
		clamp(i32(uv.x * f32(size.x)), 0, size.x - 1),
		clamp(i32(uv.y * f32(size.y)), 0, size.y - 1)
	);
	return texelFetch(vkofTextures[pc.handles.radiance], texel, 0).rgb;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 tid = gl_GlobalInvocationID.x;
	if (tid >= pc.threadCount) {
		return;
	}

	ShadingFrame frame;
	// y-up, matching the environment map's own convention
	// (environmentMapUvToDir returns y = cos(theta)), so v < 0.5 is the
	// surface's upper hemisphere and a "sun" placed high in the map is
	// actually above this surface's horizon
	frame.tanX = f32v3(1.0f, 0.0f, 0.0f);
	frame.tanY = f32v3(0.0f, 0.0f, -1.0f);
	frame.nor = f32v3(0.0f, 1.0f, 0.0f);
	const ShadingFrame coatFrame = frame;

	MaterialTableHandles tables;
	tables.kullaContyEnergy = pc.kullaContyEnergyHandle;
	tables.zeltnerLtcParam = pc.zeltnerLtcParamHandle;

	const OpenPbrMaterial mat = (
		envDlMaterial(
			pc.baseColor, pc.baseMetalness, pc.specularRoughness,
			pc.specularIor, pc.coatWeight, pc.coatRoughness, pc.coatIor
		)
	);
	const f32v3 wi = normalize(pc.wi);
	const f32 dispersedIor = openPbrTransmissionIor(mat);

	const u32 h0 = fnPcgHash(tid + fnPcgHash(pc.seedSalt));
	const u32 h1 = fnPcgHash(h0);
	u64 state = (u64(h0) << 32) | u64(h1);

	f32v3 sum = f32v3(0.0f);
	f32v3 sumSq = f32v3(0.0f);
	f32 sumLuminance = 0.0f;
	f32 sumSqLuminance = 0.0f;
	u32 bad = 0u;

	for (u32 s = 0u; s < pc.samplesPerThread; ++s) {
		f32v3 estimate = f32v3(0.0f);

		// -- light half; mirrors util-raytrace.glsl:319-386 minus the
		// shadow ray and the vdb/fog transmittance (nothing occludes here)
		if (pc.mode == ENV_DL_MODE_NEE || pc.mode == ENV_DL_MODE_MIS) {
			const f32v2 xi = (
				f32v2(fnSampleUniform(state), fnSampleUniform(state))
			);
			f32 envPdf;
			const f32v3 envDir = (
				environmentMapImportanceSample(
					pc.handles, xi, pc.rotation, envPdf
				)
			);
			if (envPdf > 0.0f) {
				const f32v3 f = (
					openPbrEvaluateF(
						tables, frame, coatFrame, wi, envDir, mat,
						/*isInsideMedium=*/false, dispersedIor
					)
				);
				if (any((f > f32v3(0.0f)))) {
					f32 weightL = 1.0f;
					if (pc.mode == ENV_DL_MODE_MIS) {
						const f32 bsdfPdfAtEnvDir = (
							openPbrEvaluatePdf(
								tables, mat, frame, coatFrame, wi, envDir,
								/*isInsideMedium=*/false, dispersedIor
							)
						);
						weightL = misBalanceWeight(envPdf, bsdfPdfAtEnvDir);
					}
					const f32v3 radiance = fnEnvRadiance(envDir);
					estimate += (
						f * abs(dot(frame.nor, envDir)) * radiance
						* weightL / envPdf
					);
				}
			}
		}

		// -- bsdf half; mirrors the throughput update at
		// util-raytrace.glsl:536 followed by the env escape at :1515,
		// including the two termination guards that precede it
		if (pc.mode == ENV_DL_MODE_BSDF || pc.mode == ENV_DL_MODE_MIS) {
			f32 bsdfPdf;
			bool sampledTransmission;
			bool sampledSubsurfaceEntry;
			u32 pickedLobe;
			const f32v3 wo = (
				openPbrSampleWo(
					tables, frame, coatFrame, wi, mat,
					bsdfPdf, sampledTransmission, sampledSubsurfaceEntry,
					pickedLobe, state,
					/*isInsideMedium=*/false, dispersedIor
				)
			);
			const f32 dotNorWo = dot(frame.nor, wo);
			const bool crossedSurface = (
				sampledTransmission
				|| (pickedLobe == OPENPBR_LOBE_SUBSURFACE_ENTRY && dotNorWo < 0.0f)
			);
			if ((crossedSurface || dotNorWo > 0.0f) && bsdfPdf > 0.0f) {
				const f32v3 f = (
					openPbrEvaluateF(
						tables, frame, coatFrame, wi, wo, mat,
						/*isInsideMedium=*/false, dispersedIor
					)
				);
				f32 weightB = 1.0f;
				if (pc.mode == ENV_DL_MODE_MIS) {
					const f32 envPdfAtWo = (
						environmentMapPdf(pc.handles, wo, pc.rotation)
					);
					weightB = misBalanceWeight(bsdfPdf, envPdfAtWo);
				}
				const f32v3 radiance = fnEnvRadiance(wo);
				estimate += (
					f * abs(dotNorWo) / bsdfPdf * radiance * weightB
				);
			}
		}

		if (any(isnan(estimate)) || any(isinf(estimate))) {
			bad += 1u;
			continue;
		}
		f32 estimateLuminance = (
			dot(estimate, f32v3(0.2126f, 0.7152f, 0.0722f))
		);
		// mirrors resolve.comp's fnFireflyClamp: rescale by the luminance
		// ratio so a clamped sample keeps its hue
		if (
			pc.fireflyClampLuminance > 0.0f
			&& estimateLuminance > pc.fireflyClampLuminance
		) {
			estimate *= pc.fireflyClampLuminance / estimateLuminance;
			estimateLuminance = pc.fireflyClampLuminance;
		}
		sum += estimate;
		sumSq += estimate * estimate;
		sumLuminance += estimateLuminance;
		sumSqLuminance += estimateLuminance * estimateLuminance;
	}

	((f32 *)(pc.sumVa))[tid * 4u + 0u] = sum.x;
	((f32 *)(pc.sumVa))[tid * 4u + 1u] = sum.y;
	((f32 *)(pc.sumVa))[tid * 4u + 2u] = sum.z;
	((f32 *)(pc.sumVa))[tid * 4u + 3u] = sumLuminance;
	((f32 *)(pc.sumSqVa))[tid * 4u + 0u] = sumSq.x;
	((f32 *)(pc.sumSqVa))[tid * 4u + 1u] = sumSq.y;
	((f32 *)(pc.sumSqVa))[tid * 4u + 2u] = sumSq.z;
	((f32 *)(pc.sumSqVa))[tid * 4u + 3u] = sumSqLuminance;
	if (bad != 0u) {
		atomicAdd(&((u32 *)(pc.badVa))[0], bad);
	}
}
