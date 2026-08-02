module "util-raytrace";
__exported import "util-types";
__exported import "util-mesh-raytrace";
__exported import "util-shading-frame";
__exported import "util-vdb";
__exported import "util-blackbody";
__exported import "util-material-openpbr";
__exported import "util-material-openpbr-load";
__exported import "util-emitter";
__exported import "util-mis";
#include "util-nan-probe.slang"
__exported import "environment";
__exported import "util-global-pc";
__exported import "util-environment-map";
// util to trace rays in the scene using ray queries



// (TODO REVIEW)
// (TODO REVIEW)

// (TODO REVIEW)
// were reached through the includer; a slang module cannot see that scope

// set per invocation by utilRaytraceBind(); was pc.global.models
public static u64 gUtilRaytraceModels = u64(0);

public func utilRaytraceBind(
	models: u64,
) {
	gUtilRaytraceModels = models;
}
// (TODO REVIEW)

// -----------------------------------------------------------------------------
// -- ray-query utility
// -----------------------------------------------------------------------------

public struct RayQueryResult {
	public f32 dist;
	public u32 modelDrawIndex;
	public u32 primitiveIndex;
	public f32v2 barycentric;
};

// (TODO REVIEW)
// openpbr geometry_opacity resolved to a coverage fraction: the alpha in
// M_pbr = mix(S_ambient_medium, M_surface, alpha). gltf MASK quantizes it
// to 0/1 (spec: opaque when alpha >= cutoff), BLEND keeps it fractional,
// OPAQUE arrives already forced to 1 by mor's loader
public func utilAlphaCoverage(
	mat: OpenPbrMaterial,
) -> f32 {
	if (mat.alphaMode == MOR_ALPHA_MODE_MASK) {
		return mat.geometryOpacity >= mat.alphaCutoff ? 1.0f : 0.0f;
	}
	return mat.geometryOpacity;
}

// stochasticCoverage: BLEND surfaces pass the ray straight through with
// probability 1-alpha instead of thresholding. unbiased, and because it
// resolves inside traversal the pass-through costs no bounce and no re-trace
public func utilTraceRayImpl(
	origin: f32v3,
	dir: f32v3,
	maxDist: f32,
	isShadowRay: bool,
	testAlphaCutoff: bool,
	stochasticCoverage: bool,
	inout state: u64,
) -> RayQueryResult {
	RayQuery<RAY_FLAG_NONE> rq;
	u32 rayFlags = 0;
	if (!testAlphaCutoff) {
		rayFlags |= RAY_FLAG_FORCE_OPAQUE;
	}
	if (isShadowRay) {
		rayFlags |= RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
	}
	RayDesc rqRay;
	rqRay.Origin = origin;
	rqRay.TMin = 0.00001f;
	rqRay.Direction = dir;
	rqRay.TMax = maxDist;
	rq.TraceRayInline(vkofTlas, rayFlags, 0xFF, rqRay);

	while (rq.Proceed()) {
		if (
			rq.CandidateType()
			!= CANDIDATE_NON_OPAQUE_TRIANGLE
		) {
			// procedural AABB candidates are unsupported; leave unconfirmed
			continue;
		}

		UtilMeshAttributeDataFromIndices rqData = (
			utilMeshAttributeDataFromIndices(
				gUtilRaytraceModels,
				rq.CandidateInstanceID(),
				rq.CandidatePrimitiveIndex(),
				rq.CandidateTriangleBarycentrics(),
				/*unpackUvDerivatives=*/true
			)
		);
		f32v3 modelNormal;
		f32v4 tangentNormal;
		f32v3 modelCoatNormal;
		f32v4 modelClearcoatNormal;
		const OpenPbrMaterial mat = (
			openPbrLoadMaterialDeriv(
				/*materialBuf=*/ rqData.materialBuf,
				/*materialIdx=*/ rqData.materialIndex,
				/*uvTransforms=*/rqData.uvTransforms,
				/*uv=*/rqData.uv,
				/*uvDx=*/rqData.uvDx,
				/*uvDy=*/rqData.uvDy,
				modelNormal, tangentNormal, modelCoatNormal, modelClearcoatNormal
			)
		);

		// a fresh draw per candidate, never one shared along the ray:
		// two stacked alpha-0.5 cards must pass 0.25 of the time, not 0.5
		const f32 coverage = utilAlphaCoverage(mat);
		const bool covered = (
			stochasticCoverage
			? fnSampleUniform(state) < coverage
			: coverage >= 0.5f
		);
		if (covered) {
			rq.CommitNonOpaqueTriangleHit();
			if (isShadowRay) {
				rq.Abort();
			}
		}
	}

	// read back the committed intersection; candidates can arrive in any
	// order, so the last candidate examined is not necessarily the closest
	if (
		rq.CommittedStatus()
		== COMMITTED_TRIANGLE_HIT
	) {
		RayQueryResult result;
		result.dist = rq.CommittedRayT();
		result.modelDrawIndex = (
			rq.CommittedInstanceID()
		);
		result.primitiveIndex = (
			rq.CommittedPrimitiveIndex()
		);
		result.barycentric = (
			rq.CommittedTriangleBarycentrics()
		);
		return result;
	}

	return RayQueryResult(0.0f, 0u, 0u, f32v2(0.0f));
}

// path-tracing entry: BLEND coverage resolves stochastically against the
// path's rng stream
public func utilTraceRay(
	origin: f32v3,
	dir: f32v3,
	maxDist: f32,
	isShadowRay: bool,
	testAlphaCutoff: bool,
	inout state: u64,
) -> RayQueryResult {
	return utilTraceRayImpl(
		origin, dir, maxDist, isShadowRay, testAlphaCutoff,
		/*stochasticCoverage=*/true, state
	);
}

// aov/picking/debug entry: no rng stream to spend, and a stochastic hit
// would make picking and single-sample probes flicker frame to frame, so
// BLEND collapses to a half-coverage threshold
public func utilTraceRay(
	origin: f32v3,
	dir: f32v3,
	maxDist: f32,
	isShadowRay: bool,
	testAlphaCutoff: bool,
) -> RayQueryResult {
	u64 unusedState = 0ul;
	return utilTraceRayImpl(
		origin, dir, maxDist, isShadowRay, testAlphaCutoff,
		/*stochasticCoverage=*/false, unusedState
	);
}
// (TODO REVIEW)


// (TODO REVIEW)
// -----------------------------------------------------------------------------
// -- emissive-mesh next event estimation
// -----------------------------------------------------------------------------

// solid-angle density the emitter strategy would have had for a bsdf-sampled
// ray that landed on an emissive triangle; 0 when that triangle is not in the
// list, which leaves the bsdf technique at full weight
public func utilEmitterHitPdf(
	unf: GpuGlobalExtended,
	hitModelDrawIndex: u32,
	hitPrimitiveIndex: u32,
	fromPosition: f32v3,
	direction: f32v3,
	hitDistance: f32,
) -> f32 {
	if (unf.meshNeeEnabled == 0u || unf.emitterLightCount == 0u) {
		return 0.0f;
	}
	GpuEmitterLight * lights = (GpuEmitterLight *)(unf.emitterLights);
	GpuResolveModelIndirect * models = (
		(GpuResolveModelIndirect *)(gUtilRaytraceModels)
	);
	const GpuResolveModelIndirect model = models[hitModelDrawIndex];
	const EmitterHitLookup lookup = (
		utilEmitterFindForHit(
			lights, model.emitterLightOffset, model.emitterLightCount,
			hitPrimitiveIndex
		)
	);
	if (!lookup.found) {
		return 0.0f;
	}
	return (
		utilEmitterPdf(
			lights, lookup.lightIndex, model, hitPrimitiveIndex,
			fromPosition, direction, hitDistance
		)
	);
}

// (TODO REVIEW)
// emitted radiance at a sampled emitter point, read from that point's own
// material/uv instead of the light list's precomputed value.
//
// the list's `radiance` is the material factors alone -- emitter-tables.cpp
// ignores emissive textures on purpose, so `power` stays cheap enough to
// rebuild every frame. that is fine for steering the pick and wrong as
// radiance: on a textured emitter it makes nee and the bsdf-hit branch
// disagree about how bright the same point is, which no mis weight can repair
// (plan-nee.md section 3.2, decision 5 -- the deliberate divergence from cull,
// whose sampler already carried primitiveIndex/barycentric for exactly this)
public func utilEmitterSampleRadiance(
	lights: GpuEmitterLight *,
	es: EmitterSample,
) -> f32v3 {
	// glsl parameters are mutable copies; slang's are not
	var lights = lights;
	UtilMeshAttributeDataFromIndices data = (
		utilMeshAttributeDataFromIndices(
			gUtilRaytraceModels,
			lights[es.lightIndex].modelDrawIndex,
			es.primitiveIndex,
			// EmitterSample carries (b0, b1, b2); the mesh helper takes the
			// ray-query convention, (b1, b2) with b0 implied
			f32v2(es.barycentric.y, es.barycentric.z),
			/*unpackUvDerivatives=*/false
		)
	);
	f32v3 emitterNormal;
	f32v4 emitterTangent;
	f32v3 emitterCoatNormal;
	f32v4 emitterCoatTangent;
	const OpenPbrMaterial emitterMaterial = (
		openPbrLoadMaterialDeriv(
			/*materialBuf=*/ data.materialBuf,
			/*materialIdx=*/ data.materialIndex,
			/*uvTransforms=*/data.uvTransforms,
			/*uv=*/data.uv,
			/*uvDx=*/f32v2(0.0f),
			/*uvDy=*/f32v2(0.0f),
			emitterNormal, emitterTangent, emitterCoatNormal, emitterCoatTangent
		)
	);
	return emitterMaterial.emissionColor * emitterMaterial.emissionLuminance;
}
// (TODO REVIEW)

// one draw over the combined light list. the environment map sits in that
// list as a non-mesh entry (emitter-tables.cpp), so a single pick and a
// single shadow ray serve both techniques.
//
// shared by all three nee sites -- surface vertex, subsurface exit vertex,
// vdb medium vertex -- because the failure trap 2 of plan-nee.md describes is
// exactly two sites disagreeing about which entry carries which pick
// probability, and a shared struct is the only way that cannot happen
public struct LightPick {
	// the list technique ran at all. the next vertex's mesh-side mis weight
	// has something to balance against exactly when this is true -- note it
	// records that a *pick* happened, not which entry it landed on: mis
	// balances densities, and the mesh entries' density for a direction is
	// nonzero however this one draw happened to fall
	public bool sampledLights;
	// the draw landed on a mesh entry and mesh nee is on
	public bool meshPicked;
	public EmitterPick pick;
	// run the env branch: the draw landed on the env entry, or the env is not
	// in the list at all and so still owns the whole pick mass
	public bool envPicked;
	// true when the env branch may reuse this pick's draws; false when it has
	// to take its own, which is what keeps an env-only scene's rng stream
	// identical to the pre-list behaviour
	public bool envFromList;
	// the env entry's share of the pick *regardless* of what this draw hit.
	// every env-side mis weight in the path uses this, forward and backward,
	// or the two halves price the same direction differently
	public f32 envPickPdf;
	// (triangle pick, barycentric u, barycentric v) for the mesh branch
	public f32v3 xi;
};

// the env entry's share of the pick, without drawing anything -- the main
// loop needs it for a vertex it did not itself pick at (a vdb medium scatter)
public func utilLightEnvPickPdf(
	unf: GpuGlobalExtended,
	envNeeActive: bool,
) -> f32 {
	if (!envNeeActive) {
		return 0.0f;
	}
	if (unf.envLightIndex >= unf.emitterLightCount) {
		return 1.0f;
	}
	return (
		utilEmitterPickPdf(
			(GpuEmitterLight *)(unf.emitterLights), unf.envLightIndex
		)
	);
}

public func utilLightPick(
	unf: GpuGlobalExtended,
	envNeeActive: bool,
	inout state: u64,
) -> LightPick {
	LightPick lp;
	lp.sampledLights = unf.emitterLightCount > 0u;
	lp.meshPicked = false;
	lp.pick.valid = false;
	lp.pick.lightIndex = 0u;
	lp.pick.pickPdf = 0.0f;
	lp.xi = f32v3(0.0f);
	lp.envPickPdf = utilLightEnvPickPdf(unf, envNeeActive);

	const bool envInList = (
		envNeeActive && unf.envLightIndex < unf.emitterLightCount
	);
	if (unf.emitterLightCount > 0u) {
		// all four drawn together so the stream length does not depend on
		// which branch the pick lands in
		const f32v4 xi4 = f32v4(
			fnSampleUniform(state), fnSampleUniform(state),
			fnSampleUniform(state), fnSampleUniform(state)
		);
		lp.xi = f32v3(xi4.y, xi4.z, xi4.w);
		lp.pick = (
			utilEmitterPick(
				(GpuEmitterLight *)(unf.emitterLights),
				unf.emitterLightCount, xi4.x
			)
		);
	}
	lp.envFromList = (
		envInList && lp.pick.valid && lp.pick.lightIndex == unf.envLightIndex
	);
	lp.envPicked = envNeeActive && (lp.envFromList || !envInList);
	lp.meshPicked = (
		unf.meshNeeEnabled != 0u && lp.pick.valid && !lp.envFromList
	);
	return lp;
}
// (TODO REVIEW)

// (TODO REVIEW)
// emissive-nee instrumentation. compiled out entirely unless the shader is
// built with EMITTER_NEE_DEBUG, which only the unit tests do -- it exists to
// tell an nee contribution that was never attempted apart from one the shadow
// ray rejected, which the rendered image alone cannot distinguish
#ifdef EMITTER_NEE_DEBUG
#define EMITTER_NEE_COUNT(unf, slot) \
	if ((unf).emitterNeeDebug != 0ul) { \
		atomicAdd(&((u32 *)((unf).emitterNeeDebug))[slot], 1u); \
	}
#else
#define EMITTER_NEE_COUNT(unf, slot)
#endif
// (TODO REVIEW)

// -----------------------------------------------------------------------------
// -- irradiance propagation
// -----------------------------------------------------------------------------

// no contributions were found. It's still okay to continue bouncing as well
// it's okay to use the accumulated irradiance. If there's no accumulated
// irradiance and the path is terminated, then no overall contribution
public static const u32 skHitNone = 0u;
// an indirect contribution was found. The path can continue bouncing, and
// on path termination the accumulated irradiance can be used.
public static const u32 skHitIndirect = 1u;
// the path should be terminated. Any contribution can be used however
// if there is none accumulated, then no overall contribution.
public static const u32 skHitTerminate = 2u;
// (TODO REVIEW)
// view-preview cut: the drag preview's second segment exists so refracted and
// mirror rays don't die inside glass, and past this roughness that segment
// can't carry a recognizable image -- not worth a full-screen trace per frame
public static const f32 skViewPreviewSmoothRoughness = 0.1f;
// (TODO REVIEW)

public struct Ray {
	public f32v3 origin;
	public f32v3 dir;
};

// forward declaration: utilIrradiance hands off to this when
// openPbrSampleWo picks the subsurface-entry lobe; defined below
// utilLoadSurfaceHit (util-raytrace.glsl further down), which its exit
// vertex reuses
bool utilSubsurfaceWalk(
	const MaterialTableHandles tables,
	inout f32v3 itOri,
	inout f32v3 bsdfWo,
	const OpenPbrMaterial entryMaterial,
	// (TODO REVIEW)
	// outward normal at the entry vertex; the dwivedi guiding axis
	const f32v3 entryNormal,
	// (TODO REVIEW)
	inout f32v3 irradianceThroughput,
	inout f32v3 irradianceAccumulator,
	inout u64 state,
	const GpuGlobalExtended unf,
	const EnvironmentMapHandles envHandles,
	const bool envNeeActive,
	// pdf of the exit-vertex refraction that produced the returned bsdfWo;
	// the caller's env-miss mis weighting needs this, not the entry pdf
	out f32 outExitPdf
);

public func utilIrradiance(
	tables: MaterialTableHandles,
	inout vdb: UtilVdb,
	material: OpenPbrMaterial,
	inout itOri: f32v3,
	inout itFrame: ShadingFrame,
	inout itCoatFrame: ShadingFrame,
	inout itNormalGeometrical: f32v3,
	inout itWi: f32v3,
	inout irradianceThroughput: f32v3,
	inout irradianceAccumulator: f32v3,
	out bsdfWo: f32v3,
	out bsdfPdf: f32,
	inout state: u64,
	out sampledTransmission: bool,
	iteration: i32,
	isInsideMedium: bool,
	inout pathSpectralChannel: i32,
	unf: GpuGlobalExtended,
	envHandles: EnvironmentMapHandles,
	envNeeActive: bool,
	hitModelDrawIndex: u32,
	hitPrimitiveIndex: u32,
	hitDistance: f32,
	hitFromPosition: f32v3,
	previousSamplePdf: f32,
	previousBounceWasDelta: bool,
	previousVertexSampledLights: bool,
	out vertexSampledLights: bool,
	out vertexEnvPickPdf: f32,
	out sampledDelta: bool,
) -> i32 {
	i32 returnEnum = skHitNone;
	// (TODO REVIEW)
	vertexSampledLights = false;
	vertexEnvPickPdf = 0.0f;
	sampledDelta = false;
	const bool meshNeeActive = (
		unf.meshNeeEnabled != 0u && unf.emitterLightCount > 0u
	);
	// (TODO REVIEW)

	// -- surface emission at this vertex, weighted by the throughput carried
	// into it; plain bsdf path tracing, every vertex accumulates its own
	// emission
	const f32v3 emission = material.emissionColor * material.emissionLuminance;
	if (any((emission > f32v3(0.0f)))) {
		// (TODO REVIEW)
		// balance against the light-sampling technique's density for this same
		// triangle. weight 1 whenever nee could not have produced this path:
		// a delta bounce, or a previous vertex that ran no light sampling
		f32 weightB = 1.0f;
		if (meshNeeActive && previousVertexSampledLights && !previousBounceWasDelta) {
			const f32 lightPdf = (
				utilEmitterHitPdf(
					unf, hitModelDrawIndex, hitPrimitiveIndex, hitFromPosition,
					normalize(itOri - hitFromPosition), hitDistance
				)
			);
			weightB = misBalanceWeight(previousSamplePdf, lightPdf);
		}
		irradianceAccumulator += emission * irradianceThroughput * weightB;
		// (TODO REVIEW)
		returnEnum = skHitIndirect;
	}

	// -- dispersion: resolve this bounce's transmission ior. discretized
	// hero-wavelength dispersion (wilkie, nawaz, droske, weidlich & hanika
	// 2014, "hero wavelength spectral sampling", \S3): the paper samples
	// one continuous hero wavelength per path and converts the resulting
	// monochromatic radiance to rgb via cie matching functions. this
	// renderer has no spectral upsampling of textures/lights elsewhere
	// (materials are plain rgb), so a continuous spectrum has nothing to
	// buy here; instead the hero wavelength is collapsed to exactly the
	// three fraunhofer lines the abbe number is already calibrated
	// against (openPbrDispersionIorRgb), i.e. a 3-sample discretization of
	// the same idea rather than the paper's continuous/MIS'd variant
	/*
		\begin{align*}
		&\textbf{(1) channel pick, once per path} \\
		&c \sim \mathrm{Uniform}\{R, G, B\}, \quad p(c) = 1/3 \\
		&\textbf{(2) unbiased per-channel estimator} \\
		% only channel c's throughput carries this path's contribution from
		% here on; the 1/p(c) = 3 weight keeps E[\cdot] correct per channel,
		% same as any discrete stochastic split (russian roulette above
		% does the identical 1/p compensation for path termination)
		&\hat{L}_c = \frac{\mathbb{1}[\text{channel} = c]}{p(c)} L_c = 3 L_c
			\tag{wilkie et al. 2014, discretized}\\
		\end{align*}
	*/
	// a photon keeps one wavelength for its whole life, so once locked
	// (pathSpectralChannel >= 0) reuse it; otherwise provisionally draw a
	// channel now in case this bounce turns out to sample transmission
	// through a dispersive material below. the draw is only committed
	// (and the throughput only channel-locked) after sampledTransmission
	// comes back -- picking a different lobe this bounce just discards it
	const bool dispersionAlreadyLocked = pathSpectralChannel >= 0;
	// c ~ Uniform{R, G, B}
	const i32 provisionalChannel = (
		dispersionAlreadyLocked
			? pathSpectralChannel
			: min(i32(fnSampleUniform(state) * 3.0f), 2)
	);
	// n(\lambda_c); falls back to the flat interface ior when this
	// material has no dispersion at all
	// (TODO REVIEW)
	// both branches carry specular_weight's ior modulation so the refraction
	// matches the reflection lobe's \eta'_s
	const f32 dispersedIor = (
		material.transmissionDispersionScale > 0.0f
			? openPbrDispersionIorRgb(material)[provisionalChannel]
			: openPbrTransmissionIor(material)
	);
	// (TODO REVIEW)

	// (TODO REVIEW)
	// -- one power-weighted pick over the combined light list, shared with the
	// subsurface-exit and vdb-medium sites (utilLightPick)
	GpuEmitterLight * neeLights = (
		(GpuEmitterLight *)(unf.emitterLights)
	);
	const LightPick lightPick = utilLightPick(unf, envNeeActive, state);
	vertexEnvPickPdf = lightPick.envPickPdf;
	// (TODO REVIEW)

	// -- next event estimation: importance-samples the env map directly
	// instead of relying on a bsdf-sampled bounce to hit it by chance,
	// mis-combined with the bsdf-sampled technique below (see the bounce-
	// miss block's weightB). uses the incoming irradianceThroughput --
	// before this bounce's own bsdf f*cos/pdf update -- since this is a
	// same-vertex direct-lighting estimate, not part of the indirect
	// continuation. dispersedIor is already resolved above and reused as
	// -is; nee never calls openPbrSampleWo, so it can't trigger the
	// dispersion provisional -> locked commit below
	if (lightPick.envPicked) {
		// (TODO REVIEW)
		// an in-list pick already spent its draws above; a standalone env nee
		// takes its own two, which keeps the stream identical to the
		// pre-combined-list behaviour for an env-only scene
		const f32v2 xi = (
			lightPick.envFromList
				? f32v2(lightPick.xi.x, lightPick.xi.y)
				: f32v2(fnSampleUniform(state), fnSampleUniform(state))
		);
		// this technique's real solid-angle density is pick * map, and every
		// weight and every division below has to use that product
		const f32 envPickPdf = lightPick.envPickPdf;
		// (TODO REVIEW)
		f32 envPdf;
		const f32v3 envDir = (
			environmentMapImportanceSample(
				envHandles, xi, unf.envRotation, envPdf
			)
		);
		// (TODO REVIEW)
		const f32 envLightPdf = envPickPdf * envPdf;
		if (envLightPdf > 0.0f) {
			// (TODO REVIEW)
			const f32v3 f = (
				openPbrEvaluateF(
					tables, itFrame, itCoatFrame, itWi, envDir, material,
					isInsideMedium, dispersedIor
				)
			);
			// cheap early-out before tracing a shadow ray
			if (any((f > f32v3(0.0f)))) {
				const f32 bsdfPdfAtEnvDir = (
					openPbrEvaluatePdf(
						tables, material, itFrame, itCoatFrame, itWi, envDir,
						isInsideMedium, dispersedIor
					)
				);
				const f32 weightL = misBalanceWeight(envLightPdf, bsdfPdfAtEnvDir);
				// offset toward whichever side of the surface envDir
				// actually points at (unlike the bsdf-sampled bounce below,
				// there's no sampledTransmission flag for an nee direction
				// drawn straight from the env map)
				const f32v3 shadowOrigin = (
					itOri
					+ itNormalGeometrical * (
						dot(itFrame.nor, envDir) >= 0.0f
							? 0.0005f : -0.0005f
					)
				);
				// (TODO REVIEW)
				// this vertex's bsdf continuation resolves straight against the
				// env with no AS trace, so tracing nee's shadow ray here would
				// leave the two mis'd techniques disagreeing about visibility:
				// an occluded direction would land at weightB*L instead of 0 or
				// the full unoccluded estimate
				const bool envOnlyFinalVertex = (
					unf.envMapOnlyDepth != 0u
					&& iteration + 1 >= i32(unf.envMapOnlyDepth)
				);
				bool envDirVisible = true;
				if (!envOnlyFinalVertex) {
					const RayQueryResult visibilityRq = (
						utilTraceRay(
							shadowOrigin,
							envDir,
							/*maxDist=*/999999.0f,
							/*isShadowRay=*/true,
							/*testAlphaCutoff=*/true,
							state
						)
					);
					envDirVisible = visibilityRq.dist <= 0.0f;
				}
				// (TODO REVIEW)
				if (envDirVisible) {
					const f32v3 transmittance = (
						utilVdbFogTransmittance(
							vdb, shadowOrigin, envDir,
							unf.vdbSigmaAbsorption, unf.vdbSigmaScattering,
							unf.fogThickness,
							unf.fogDistanceMax,
							// (TODO REVIEW)
							// an env connection never ends, so it traverses the
							// whole cap
							/*travelDistance=*/999999.0f,
							// (TODO REVIEW)
							state
						)
					);
					const f32v3 envRadiance = (
						environmentMapRadiance(envHandles, envDir, unf.envRotation)
						* unf.envIntensity
					);
					irradianceAccumulator += (
						irradianceThroughput * f
						* abs(dot(itFrame.nor, envDir)) * envRadiance
						* weightL / envLightPdf
						* transmittance
					);
				}
			}
		}
	}

	// (TODO REVIEW)
	// -- next event estimation toward emissive meshes; the other branch of the
	// combined pick above, with the emitter sampler in place of the env map's
	// importance sampler.
	//
	// vertexSampledLights records that the *list* technique ran here, not that
	// this particular draw landed on a mesh entry: the next vertex's mis
	// weight balances against the density the technique has for that
	// direction, which is nonzero for every mesh entry in the list however
	// this one pick happened to fall
	vertexSampledLights = lightPick.sampledLights;
	if (lightPick.meshPicked) {
		GpuResolveModelIndirect * neeModels = (
			(GpuResolveModelIndirect *)(gUtilRaytraceModels)
		);
		const EmitterSample es = (
			utilEmitterSampleFromPick(
				neeLights, lightPick.pick, neeModels, itOri, lightPick.xi
			)
		);
		EMITTER_NEE_COUNT(unf, 0u)
		if (es.valid) { EMITTER_NEE_COUNT(unf, 1u) }
		if (es.valid) {
			// abs, and no sign gate: a transmitted direction reaches an
			// emitter through the surface, and openPbrEvaluateF returns 0 on
			// its own for a direction the material cannot scatter into.
			// rejecting the back hemisphere here would make this technique's
			// real density 0 where utilEmitterHitPdf still reports nonzero,
			// so the bsdf side would be down-weighted with nothing
			// compensating. mirrors the env nee block above
			const f32 dotNorEmitter = abs(dot(itFrame.nor, es.direction));
			{
				const f32v3 fEmitter = (
					openPbrEvaluateF(
						tables, itFrame, itCoatFrame, itWi, es.direction,
						material, isInsideMedium, dispersedIor
					)
				);
				if (any((fEmitter > f32v3(0.0f)))) {
					EMITTER_NEE_COUNT(unf, 2u)
					const f32 bsdfPdfAtEmitter = (
						openPbrEvaluatePdf(
							tables, material, itFrame, itCoatFrame, itWi,
							es.direction, isInsideMedium, dispersedIor
						)
					);
					const f32v3 shadowOrigin = (
						itOri
						+ itNormalGeometrical * (
							dot(itNormalGeometrical, es.direction) >= 0.0f
								? 0.0005f : -0.0005f
						)
					);
					// the shadow ray has to aim at the sampled point from the
					// *offset* origin. es.direction/es.distance are measured
					// from the shading point, so reusing them leaves the ray
					// parallel-shifted, crossing the emitter's plane short of
					// the sample -- which a fixed end margin cannot absorb,
					// since the nudge moves the plane crossing by
					// offset/dot(dir, n_emitter). recomputed here, the ray
					// ends exactly at the sample and a relative epsilon is
					// enough. the estimator keeps es.direction: f, the pdf and
					// the mis weight all belong to the shading point
					const f32v3 shadowToLight = es.position - shadowOrigin;
					const f32 shadowMaxDist = length(shadowToLight) * (1.0f - 1e-4f);
					const f32v3 shadowDir = (
						shadowMaxDist > 0.0f
							? shadowToLight / length(shadowToLight)
							: es.direction
					);
					// a segment this short can't establish visibility, and
					// es.pdf ~ distance^2 makes 1/pdf explode there, so drop
					// the sample rather than admit an unshadowed firefly
					if (shadowMaxDist > 0.0f) {
						const RayQueryResult emitterRq = (
							utilTraceRay(
								shadowOrigin,
								shadowDir,
								/*maxDist=*/ shadowMaxDist,
								/*isShadowRay=*/true,
								/*testAlphaCutoff=*/true,
								state
							)
						);
						if (emitterRq.dist > 0.0f) {
							EMITTER_NEE_COUNT(unf, 3u)
#ifdef EMITTER_NEE_DEBUG
							// split the occluder: an emissive blocker is the
							// light's own geometry getting in the way, anything
							// else is a genuine shadow
							const EmitterHitLookup blockerLookup = (
								utilEmitterFindForHit(
									neeLights,
									neeModels[
										emitterRq.modelDrawIndex
									].emitterLightOffset,
									neeModels[
										emitterRq.modelDrawIndex
									].emitterLightCount,
									emitterRq.primitiveIndex
								)
							);
							if (blockerLookup.found) {
								EMITTER_NEE_COUNT(unf, 5u)
							} else {
								EMITTER_NEE_COUNT(unf, 6u)
							}
#endif
						}
						if (emitterRq.dist <= 0.0f) {
							EMITTER_NEE_COUNT(unf, 4u)
							const f32v3 transmittance = (
								utilVdbFogTransmittance(
									vdb, shadowOrigin, es.direction,
									unf.vdbSigmaAbsorption, unf.vdbSigmaScattering,
									unf.fogThickness, unf.fogDistanceMax,
									// (TODO REVIEW)
									// the connection stops at the emitter, so only
									// that much medium is between them
									/*travelDistance=*/ shadowMaxDist,
									// (TODO REVIEW)
									state
								)
							);
							const f32v3 emitterRadiance = (
								utilEmitterSampleRadiance(
									(GpuEmitterLight *)(unf.emitterLights), es
								)
							);
							irradianceAccumulator += (
								irradianceThroughput * fEmitter * dotNorEmitter
								* emitterRadiance * transmittance
								* misBalanceWeight(es.pdf, bsdfPdfAtEmitter) / es.pdf
							);
						}
					}
				}
			}
		}
	}
	// (TODO REVIEW)

	// -- propagate irradiance throughput along bsdf
	u32 pickedLobe;
	bool sampledSubsurfaceEntry;
	bsdfWo = (
		openPbrSampleWo(
			/*tables=*/ tables,
			/*frame=*/ itFrame,
			/*coatFrame=*/ itCoatFrame,
			/*wi=*/ itWi,
			/*mat=*/ material,
			/*pdf=*/ bsdfPdf,
			/*sampledTransmission=*/ sampledTransmission,
			/*sampledSubsurfaceEntry=*/ sampledSubsurfaceEntry,
			/*pickedLobe=*/ pickedLobe,
			state,
			/*isInsideMedium=*/ isInsideMedium,
			/*dispersedIor=*/ dispersedIor
		)
	);
	// \hat{L}_c = 3 L_c: commit the provisional pick only on the unlocked
	// -> locked transition, and only compensate throughput once for it
	if (
		!dispersionAlreadyLocked
		&& sampledTransmission
		&& material.transmissionDispersionScale > 0.0f
	) {
		pathSpectralChannel = provisionalChannel;
		f32v3 channelMask = f32v3(0.0f);
		channelMask[provisionalChannel] = 3.0f;
		irradianceThroughput *= channelMask;
	}

	// -- subsurface entry: hands off to the bounded random walk instead of
	// falling through to the generic f*cos/pdf update below, which would
	// incorrectly zero throughput here (openPbrSubsurfaceEvaluateF always
	// returns 0 -- this lobe has no instantaneous local value, see its
	// header comment). the walk's own exit-vertex refraction applies its
	// own fresnel-weighted throughput update internally
	if (sampledSubsurfaceEntry) {
		// the walk replaces the f*cos/pdf update, so divide out the entry
		// lobe's selection probability by hand: scaling by probabilityTotal
		// leaves the layered coefficient (p_sss * probabilityTotal ==
		// probabilitySubsurfaceEntry). applied pre-walk so its exit nee scales
		// too; recomputed deterministically to match openPbrSampleWo's draw
		const OpenPbrLobeSelection subsurfaceSel = (
			openPbrLobeSelection(tables, material, dot(itFrame.nor, itWi))
		);
		irradianceThroughput *= subsurfaceSel.probabilityTotal;
		itOri -= itNormalGeometrical * 0.000001f;
		const bool exited = (
			utilSubsurfaceWalk(
				// (TODO REVIEW)
				tables, itOri, bsdfWo, material, itFrame.nor,
				// (TODO REVIEW)
				irradianceThroughput, irradianceAccumulator,
				state, unf, envHandles, envNeeActive, bsdfPdf
			)
		);
		if (!exited) {
			return skHitTerminate;
		}
		// (TODO REVIEW)
		// the next segment leaves the walk's *exit* vertex, not this entry
		// vertex, so the path state the next vertex balances against is the
		// exit's -- and the exit runs the same combined pick this vertex did.
		// left as the entry vertex's state, an emissive triangle hit straight
		// after the exit would be balanced against a density measured at the
		// wrong point
		vertexSampledLights = unf.emitterLightCount > 0u;
		vertexEnvPickPdf = utilLightEnvPickPdf(unf, envNeeActive);
		// (TODO REVIEW)
		if (length(irradianceThroughput) < 1e-4f) {
			return skHitTerminate;
		}
		if (iteration > 2) {
			const f32 rrChanceUnclamped = (
				max(
					irradianceThroughput.x,
					max(irradianceThroughput.y, irradianceThroughput.z)
				)
			);
			const f32 rrChance = clamp(rrChanceUnclamped, 0.0f, 1.0f);
			if (fnSampleUniform(state) > rrChance) {
				return skHitTerminate;
			}
			irradianceThroughput /= rrChance;
		}
		// itOri/bsdfWo already sit at the walk's exit boundary, nudged
		// along the exit's own normal by utilSubsurfaceWalk -- unlike the
		// reflect/transmit nudge at the end of this function (keyed to
		// itNormalGeometrical, the *entry* vertex's normal, which would be
		// wrong here), so this returns directly rather than falling
		// through to that tail nudge
		return returnEnum;
	}

	// uncomment this block to test itFrame.nor vs itNormalGeometrical
#if 0
	if (dot(itNormalGeometrical, bsdfWo) < 0.0f)
	{
		// if the sampled wo is below the surface, then terminate path
		irradianceAccumulator = (itNormalGeometrical-itFrame.nor);
		return skHitTerminate;
	}
#endif

	// below-surface check: valid for reflection but not for transmission
	// (refracted wo correctly points below the surface normal)
	f32 dotNorWo = dot(itFrame.nor, bsdfWo);
	// the thin-walled subsurface sheet's diffuse transmission lobe also
	// crosses the surface, but unlike sampledTransmission it enters no
	// medium -- a thin wall has no interior, so isInsideMedium must not flip
	const bool sampledThinWalledTransmission = (
		pickedLobe == OPENPBR_LOBE_SUBSURFACE_ENTRY && dotNorWo < 0.0f
	);
	const bool crossedSurface = (
		sampledTransmission || sampledThinWalledTransmission
	);
	// (TODO REVIEW)
	// nee can never reproduce these directions, so the next vertex must weight
	// its emission at 1: a thin wall transmits along exactly wo = -wi, and
	// openPbrSampleWo's tir fallback draws the same exact reflection
	sampledDelta = (
		(sampledTransmission && material.geometryThinWalled > 0.0f)
		// a matched interface takes the same delta branch in
		// openPbrTransmission{EvaluateF,Pdf}; exact equality, because that is
		// the test those two use and the three must agree on which lobe is delta
		|| (sampledTransmission && dispersedIor == 1.0f)
		|| (
			material.specularRoughness <= 0.0f
			&& (pickedLobe == OPENPBR_LOBE_SPECULAR || sampledTransmission)
		)
	);
	// (TODO REVIEW)
	if (!crossedSurface && dotNorWo <= 0.0f) {
		return skHitTerminate;
	}

	// a sampled wo can legitimately land with zero density: openPbrSampleWo's
	// TIR fallback draws a continuous h, and wo == -wi exactly (or close
	// enough that every lobe's pdf floors to 0 there, see
	// openPbrTransmissionTirReflectPdf / utilMicrofacetGgxBoundedReflectPdfAniso's
	// zero-vector guards) is a reachable, if rare, draw. f is structurally
	// zero there too for a material with no diffuse/coat/fuzz budget (a pure
	// dielectric), so f / bsdfPdf is a genuine 0/0, not a formula bug -- treat
	// it the same as any other zero-contribution sample rather than dividing
	if (bsdfPdf <= 0.0f) {
		return skHitTerminate;
	}

	// propagate irradiance throughput along bsdf; use abs() so transmission
	// cosine factor is positive (wo is below normal for refracted rays)
	irradianceThroughput *= (
		openPbrEvaluateF(
			tables, itFrame, itCoatFrame, itWi, bsdfWo, material,
			isInsideMedium, dispersedIor
		)
		* abs(dotNorWo) / bsdfPdf
	);
	// f and bsdfPdf are individually finite (openPbrEvaluateF/openPbrSampleWo
	// are both NAN_CHECK-guarded), but f/bsdfPdf can still overflow to Inf
	// when bsdfPdf is a genuine, non-NaN near-zero -- an Inf throughput isn't
	// caught by the length()-based termination below (length(Inf) is Inf,
	// not < 1e-4f) and only turns into a NaN several bounces later when it
	// eventually multiplies against an exact-zero channel, far from this
	// call site; check here instead, at the point of overflow
	NAN_CHECK3(
		irradianceThroughput,
		"NaN utilIrradiance.irradianceThroughput after bsdf divide px(%d,%d)=%v3f"
	)

	// terminate path if throughput is too low to contribute; this also helps
	// prevent NaN accumulation
	if (length(irradianceThroughput) < 1e-4f) {
		return skHitTerminate;
	}

	// -- russian roulette
	if (iteration > 2) {
		const f32 rrChanceUnclamped = (
			max(
				irradianceThroughput.x,
				max(irradianceThroughput.y, irradianceThroughput.z)
			)
		);
		const f32 rrChance = clamp(rrChanceUnclamped, 0.0f, 1.0f);
		if (fnSampleUniform(state) > rrChance) {
			return skHitTerminate;
		}
		irradianceThroughput /= rrChance;
	}

	// -- nudge the origin off the surface for the next walk segment's trace
	// (this offset lived in the old end-of-bounce trace here): reflection
	// continues on the front side, transmission on the back side. applied
	// last so the nee shadow ray above still originates on the surface
	itOri += itNormalGeometrical * (crossedSurface ? -0.0005f : 0.0005f);

	return returnEnum;
}

public func utilIrradiancePropagate(
	tables: MaterialTableHandles,
	inout vdb: UtilVdb,
	inout material: OpenPbrMaterial,
	inout itOri: f32v3,
	inout itFrame: ShadingFrame,
	inout itCoatFrame: ShadingFrame,
	inout itNormalGeometrical: f32v3,
	inout itWi: f32v3,
	inout bsdfWo: f32v3,
	out bsdfPdf: f32,
	inout irradianceThroughput: f32v3,
	inout irradianceAccumulator: f32v3,
	inout state: u64,
	iteration: i32,
	inout isInsideMedium: bool,
	out sampledTransmission: bool,
	inout pathSpectralChannel: i32,
	unf: GpuGlobalExtended,
	envHandles: EnvironmentMapHandles,
	envNeeActive: bool,
	hitModelDrawIndex: u32,
	hitPrimitiveIndex: u32,
	hitDistance: f32,
	hitFromPosition: f32v3,
	previousSamplePdf: f32,
	previousBounceWasDelta: bool,
	previousVertexSampledLights: bool,
	out vertexSampledLights: bool,
	out vertexEnvPickPdf: f32,
	out sampledDelta: bool,
) -> i32 {
	// -- compute irradiance on surface
	i32 returnEnum = (
		utilIrradiance(
			tables,
			vdb,
			material,
			itOri,
			itFrame,
			itCoatFrame,
			itNormalGeometrical,
			itWi,
			irradianceThroughput,
			irradianceAccumulator,
			bsdfWo,
			bsdfPdf,
			state,
			sampledTransmission,
			iteration,
			isInsideMedium,
			pathSpectralChannel,
			unf,
			envHandles,
			envNeeActive,
			hitModelDrawIndex,
			hitPrimitiveIndex,
			hitDistance,
			hitFromPosition,
			previousSamplePdf,
			previousBounceWasDelta,
			previousVertexSampledLights,
			vertexSampledLights,
			vertexEnvPickPdf,
			sampledDelta
		)
	);
	// (TODO REVIEW)
	// a thin wall has no interior to enter, so it must not flip the medium
	// state -- same reasoning as sampledThinWalledTransmission above. left
	// flipped, everything seen through the wall is shaded as if submerged
	// and its specular lobe hits total internal reflection (white)
	if (sampledTransmission && material.geometryThinWalled <= 0.0f) {
		isInsideMedium = !isInsideMedium;
	}
	// (TODO REVIEW)
	// beer-lambert attenuation for the entered dielectric interior moved to
	// utilIrradianceWalk, which owns the next segment and its length

	return returnEnum;
}

// loads the material and both shading frames (base + coat) at a confirmed
// triangle hit, advances itOri to the hit point, and flips the frames to
// the incoming-ray side on a back-face hit. factored out of
// utilIrradianceWalk's surface-setup block so the subsurface walk's exit
// vertex (utilSubsurfaceWalk below) can load a hit the same way without
// duplicating the mesh-unpack/tbn/back-face-flip logic
public func utilLoadSurfaceHit(
	rq: RayQueryResult,
	inout itOri: f32v3,
	bsdfWo: f32v3,
	out material: OpenPbrMaterial,
	out itFrame: ShadingFrame,
	out itCoatFrame: ShadingFrame,
	out itNormalGeometrical: f32v3,
	out itWi: f32v3,
) {
	UtilMeshAttributeDataFromIndices rqData = (
		utilMeshAttributeDataFromIndices(
			gUtilRaytraceModels,
			rq.modelDrawIndex,
			rq.primitiveIndex,
			rq.barycentric,
			/*unpackUvDerivatives=*/true
		)
	);
	const f32m3x3 tbn = (
		utilCalculateTbnBasis(rqData.normal, rqData.tangent)
	);
	f32v3 geometryNormal;
	f32v4 geometryTangent;
	f32v3 geometryCoatNormal;
	f32v4 geometryCoatTangent;
	material = (
		openPbrLoadMaterialDeriv(
			/*materialBuf=*/ rqData.materialBuf,
			/*materialIdx=*/ rqData.materialIndex,
			/*uvTransforms=*/rqData.uvTransforms,
			/*uv=*/rqData.uv,
			/*uvDx=*/rqData.uvDx,
			/*uvDy=*/rqData.uvDy,
			geometryNormal, geometryTangent, geometryCoatNormal, geometryCoatTangent
		)
	);
	itOri = itOri + bsdfWo * rq.dist;
	// (TODO REVIEW)
	// shading frame from the normal-mapped normal + openpbr geometry_tangent,
	// which is what steers the anisotropy direction; an unbound tangent map
	// decodes to +x, so tbn maps it straight back onto the mesh/uv tangent
	itFrame = (
		shadingFrameFromTangent(
			utilShadingNormalOrGeometric(
				mul(tbn, geometryNormal), rqData.normalGeometrical
			),
			mul(tbn, geometryTangent.xyz))
	);
	// coat's own shading normal + tangent (khr_materials_clearcoat has no
	// tangent of its own, so it rides the same identity fallback)
	itCoatFrame = (
		shadingFrameFromTangent(
			utilShadingNormalOrGeometric(
				mul(tbn, geometryCoatNormal), rqData.normalGeometrical
			),
			mul(tbn, geometryCoatTangent.xyz))
	);
	// (TODO REVIEW)
	itNormalGeometrical = rqData.normalGeometrical;
	itWi = -bsdfWo;
	// back-face hit: either just refracted through, or continuing to
	// bounce inside the medium via an internal reflection; flip so the
	// next bounce sees a consistent frame regardless of how it got here
	if (dot(itWi, itNormalGeometrical) < 0.0f) {
		itFrame = shadingFrameFromTangent(-itFrame.nor, itFrame.tanX);
		itCoatFrame = (
			shadingFrameFromTangent(-itCoatFrame.nor, itCoatFrame.tanX)
		);
		itNormalGeometrical = -itNormalGeometrical;
	}
}

// -----------------------------------------------------------------------------
// -- subsurface random walk
// -----------------------------------------------------------------------------

// safety backstop only, not the primary termination -- russian roulette
// below (skSubsurfaceWalkRRWarmup) is what actually bounds expected cost
// now. this used to be the primary cap at 256, which turned out to be a
// real bias, not just a performance knob: the number of bounces needed
// scales like (object size / radius)^2, and a small subsurfaceRadius
// relative to the object (or a channel with a much smaller radiusScale
// than its siblings) can legitimately need more than 256 to converge --
// hitting the cap silently zeroed exactly that channel regardless of its
// true albedo, which is what produced a physically-backwards-looking
// subsurfaceRadiusScale ratio being needed to compensate (the short-
// mean-free-path channel got truncated to black, so the only way to get
// any of it through was enlarging its radius past where it reliably hit
// the cap). kept high enough that a well-behaved rr walk essentially
// never reaches it; only a pathological configuration (near-zero
// absorption, rr never firing) would
public static const u32 skSubsurfaceWalkMaxSteps = 4096u;
// bounces before russian roulette starts culling the walk; a handful of
// free bounces up front, matching the outer path's own "iteration > 2"
// warmup in utilIrradiance
public static const u32 skSubsurfaceWalkRRWarmup = 4u;
// runs the bounded subsurface random walk from an entry vertex (itOri,
// bsdfWo already the entry position/into-surface direction, as set by
// openPbrSampleWo's subsurface-entry branch) until it either exits through
// the medium boundary or throughput collapses to nothing (absorption is
// continuous decay across collisions, not a discrete kill -- see
// openPbrSubsurfaceWalkStep's header comment). on a successful exit,
// itOri/bsdfWo are left at the exit position/refracted-out direction --
// ordinary state for the caller to continue propagating exactly as if
// this had been a normal bounce -- and returns true. returns false once
// throughput collapses, on exceeding skSubsurfaceWalkMaxSteps, or if the
// walk ray ever fails to hit anything (a mesh with a hole in it; not
// expected for a closed subsurface solid, guarded rather than assumed).
public func utilSubsurfaceWalk(
	tables: MaterialTableHandles,
	inout itOri: f32v3,
	inout bsdfWo: f32v3,
	entryMaterial: OpenPbrMaterial,
	entryNormal: f32v3,
	inout irradianceThroughput: f32v3,
	inout irradianceAccumulator: f32v3,
	inout state: u64,
	unf: GpuGlobalExtended,
	envHandles: EnvironmentMapHandles,
	envNeeActive: bool,
	out outExitPdf: f32,
) -> bool {
	outExitPdf = 0.0f;
	f32v3 sigmaAbsorption;
	f32v3 sigmaScattering;
	openPbrSubsurfaceAlbedoToSigma(
		entryMaterial.subsurfaceColor,
		entryMaterial.subsurfaceRadius * entryMaterial.subsurfaceRadiusScale,
		entryMaterial.subsurfaceScatterAnisotropy,
		sigmaAbsorption, sigmaScattering
	);

	for (i32 step = 0; step < skSubsurfaceWalkMaxSteps; ++step) {
		const RayQueryResult rq = (
			utilTraceRay(itOri, bsdfWo, 999999.0f, false, true, state)
		);
		if (rq.dist <= 0.0f) {
			return false;
		}

		f32 outDistance;
		f32v3 outWo;
		f32v3 outWeight;
		const u32 outcome = (
			openPbrSubsurfaceWalkStep(
				sigmaAbsorption, sigmaScattering, rq.dist,
				bsdfWo, entryMaterial.subsurfaceScatterAnisotropy,
				// (TODO REVIEW)
				irradianceThroughput, entryNormal,
				// (TODO REVIEW)
				outDistance, outWo, outWeight, state
			)
		);

		if (outcome == OPENPBR_SUBSURFACE_WALK_SCATTER) {
			// outWeight already includes the albedo (sigmaScattering
			// factor) -- see openPbrSubsurfaceWalkStep's header comment
			irradianceThroughput *= outWeight;
			NAN_CHECK3(
				irradianceThroughput,
				"NaN utilSubsurfaceWalk.irradianceThroughput after scatter px(%d,%d)=%v3f"
			)
			// throughput collapses continuously here (unlike a discrete
			// absorb/kill draw), so check every step rather than only
			// after the walk returns -- otherwise a highly absorptive
			// medium burns the full skSubsurfaceWalkMaxSteps budget on a
			// path already contributing nothing
			if (length(irradianceThroughput) < 1e-4f) {
				return false;
			}
			// -- russian roulette: bounds expected cost without the bias a
			// hard step cap introduces (see skSubsurfaceWalkMaxSteps's own
			// comment) -- a slowly-decaying, near-unity-albedo path is
			// exactly the case that legitimately needs many bounces, so
			// culling it by bounce count alone (rather than by its own
			// throughput) is what breaks energy conservation for
			// low-absorption channels
			if (step >= skSubsurfaceWalkRRWarmup) {
				const f32 rrChanceUnclamped = (
					max(
						irradianceThroughput.x,
						max(irradianceThroughput.y, irradianceThroughput.z)
					)
				);
				const f32 rrChance = clamp(rrChanceUnclamped, 0.0f, 1.0f);
				if (fnSampleUniform(state) > rrChance) {
					return false;
				}
				irradianceThroughput /= rrChance;
			}
			itOri += bsdfWo * outDistance;
			bsdfWo = outWo;
			continue;
		}

		// -- OPENPBR_SUBSURFACE_WALK_EXIT: reached the medium boundary.
		// attempt the dielectric refraction-out through this same
		// interface openPbrTransmissionEvaluateF/SampleWo/Pdf already
		// model for isInsideMedium=true -- reusing that already-validated
		// math rather than writing new refraction code. coat/fuzz are not
		// consulted here, the same documented scope limit
		// openPbrEvaluateF's transmission branch already carries ("no
		// furnace-test coverage exists for coat/fuzz stacked on a
		// transmissive base")
		OpenPbrMaterial exitMaterial;
		ShadingFrame exitFrame;
		ShadingFrame exitCoatFrame;
		f32v3 exitNormalGeometrical;
		f32v3 exitWi;
		utilLoadSurfaceHit(
			rq, itOri, bsdfWo,
			exitMaterial, exitFrame, exitCoatFrame, exitNormalGeometrical, exitWi
		);

		const f32 roughness = max(exitMaterial.specularRoughness, 1e-5f);
		const f32v2 xiRefract = fnSampleUniform2(state);
		const f32v3 h = (
			utilMicrofacetSampleGgxVndf(exitFrame.nor, exitWi, roughness, xiRefract)
		);
		// isInsideMedium=true convention (see openPbrSampleWo's own
		// transmission branch): etaI is the medium's own ior, etaT is air.
		// no dispersion at the exit -- subsurface's random walk already
		// resolves a chromatic weight per step (openPbrSubsurfaceWalkStep),
		// and layering transmission's separate dispersion mechanism on
		// top of that here would be new scope, not a port of anything
		// this file already does
		// (TODO REVIEW)
		// same \eta'_s the exit interface's own reflection lobe uses
		const f32 etaI = openPbrTransmissionIor(exitMaterial);
		// (TODO REVIEW)
		const f32 etaT = 1.0f;
		const f32v3 refracted = refract(-exitWi, h, etaI / etaT);

		// -- nee from the exit vertex: same shadow-ray/mis pattern as
		// utilIrradiance's own env nee block, narrowed to just this
		// dielectric interface -- the walk already stands in for the
		// diffuse/subsurface response, so evaluating the full material
		// stack here (coat/fuzz/diffuse/another subsurface entry) would
		// double count it
		// (TODO REVIEW)
		// one pick over the combined light list, exactly as the surface vertex
		// does it: the environment and the emissive meshes compete for the same
		// draw and the same shadow ray. an empty list takes no draws at all, so
		// a scene with no emitters keeps the pre-P4 rng stream here
		const LightPick exitPick = utilLightPick(unf, envNeeActive, state);
		// (TODO REVIEW)
		if (exitPick.envPicked) {
			const f32v2 xiNee = (
				exitPick.envFromList
					? f32v2(exitPick.xi.x, exitPick.xi.y)
					: fnSampleUniform2(state)
			);
			f32 envPdf;
			const f32v3 envDir = (
				environmentMapImportanceSample(
					envHandles, xiNee, unf.envRotation, envPdf
				)
			);
			// (TODO REVIEW)
			// the light technique's real density is pick * map; see the
			// surface vertex's own envLightPdf
			envPdf *= exitPick.envPickPdf;
			// (TODO REVIEW)
			if (envPdf > 0.0f) {
				const f32v3 f = (
					openPbrTransmissionEvaluateF(
						exitMaterial, exitFrame, exitWi, envDir,
						/*isInsideMedium=*/true,
						/*dispersedIor=*/etaI
					)
				);
				if (any((f > f32v3(0.0f)))) {
					const f32 bsdfPdfAtEnvDir = (
						openPbrTransmissionPdf(
							exitMaterial, exitFrame, exitWi, envDir,
							/*isInsideMedium=*/true,
							/*dispersedIor=*/etaI
						)
					);
					const f32 weightL = misBalanceWeight(envPdf, bsdfPdfAtEnvDir);
					const f32v3 shadowOrigin = (
						itOri + exitNormalGeometrical * -0.0005f
					);
					// (TODO REVIEW)
					// depth 1 puts the entry vertex at the end of the path, so
					// this exit's continuation resolves against the env
					// unoccluded -- its nee must agree (see utilIrradiance's
					// envOnlyFinalVertex). deeper env-only modes can't tell
					// which iteration entered the walk, so they keep tracing
					bool envDirVisible = true;
					if (unf.envMapOnlyDepth != 1u) {
						const RayQueryResult visibilityRq = (
							utilTraceRay(
								shadowOrigin, envDir, /*maxDist=*/999999.0f,
								/*isShadowRay=*/true, /*testAlphaCutoff=*/true,
								state
							)
						);
						envDirVisible = visibilityRq.dist <= 0.0f;
					}
					// (TODO REVIEW)
					if (envDirVisible) {
						const f32v3 envRadiance = (
							environmentMapRadiance(
								envHandles, envDir, unf.envRotation
							)
							* unf.envIntensity
						);
						irradianceAccumulator += (
							irradianceThroughput * f
							* abs(dot(exitFrame.nor, envDir)) * envRadiance
							* weightL / envPdf
						);
					}
				}
			}
		}

		// (TODO REVIEW)
		// -- the mesh branch of the same pick. the walk stands in for the
		// diffuse/subsurface response, so as with the env branch only this
		// dielectric interface's transmission lobe is evaluated -- running the
		// whole material stack here would double count the walk
		if (exitPick.meshPicked) {
			const EmitterSample es = (
				utilEmitterSampleFromPick(
					(GpuEmitterLight *)(unf.emitterLights),
					exitPick.pick,
					(GpuResolveModelIndirect *)(gUtilRaytraceModels),
					itOri,
					exitPick.xi
				)
			);
			if (es.valid) {
				const f32v3 fEmitter = (
					openPbrTransmissionEvaluateF(
						exitMaterial, exitFrame, exitWi, es.direction,
						/*isInsideMedium=*/true, /*dispersedIor=*/etaI
					)
				);
				if (any((fEmitter > f32v3(0.0f)))) {
					const f32 bsdfPdfAtEmitter = (
						openPbrTransmissionPdf(
							exitMaterial, exitFrame, exitWi, es.direction,
							/*isInsideMedium=*/true, /*dispersedIor=*/etaI
						)
					);
					// the exit's outward side, matching the env branch's own
					// offset: exitNormalGeometrical points inward
					const f32v3 shadowOrigin = (
						itOri + exitNormalGeometrical * -0.0005f
					);
					// derived from the *offset* origin, never from
					// es.direction/es.distance -- see section 3.7 of
					// plan-nee.md, where reusing them silently discarded every
					// grazing connection as self-shadowed
					const f32v3 shadowToLight = es.position - shadowOrigin;
					const f32 shadowLength = length(shadowToLight);
					if (shadowLength > 0.0f) {
						const RayQueryResult emitterRq = (
							utilTraceRay(
								shadowOrigin,
								shadowToLight / shadowLength,
								/*maxDist=*/ shadowLength * (1.0f - 1e-4f),
								/*isShadowRay=*/true,
								/*testAlphaCutoff=*/true,
								state
							)
						);
						if (emitterRq.dist <= 0.0f) {
							const f32v3 emitterRadiance = (
								utilEmitterSampleRadiance(
									(GpuEmitterLight *)(unf.emitterLights), es
								)
							);
							irradianceAccumulator += (
								irradianceThroughput * fEmitter
								* abs(dot(exitFrame.nor, es.direction))
								* emitterRadiance
								* misBalanceWeight(es.pdf, bsdfPdfAtEmitter)
								/ es.pdf
							);
						}
					}
				}
			}
		}
		// (TODO REVIEW)

		// (TODO REVIEW)
		// total internal reflection: stays inside, walk continues from this
		// same boundary point back into the medium. exitNormalGeometrical
		// points inward (utilLoadSurfaceHit's back-face flip, same convention
		// utilIrradiance's own tail nudge relies on), so continuing inside
		// nudges along +normal.
		//
		// this sits *below* the nee block deliberately. tir is a property of
		// the one stochastically sampled microfacet h, not of the vertex: the
		// analytic half-vector toward the emitter can transmit while this h
		// does not. skipping nee here gated the light technique on the bsdf
		// sample's outcome, dropping that vertex's direct light while the bsdf
		// side stayed mis-down-weighted at full rate -- a deficit that grew
		// with roughness, since a wider slope distribution tirs more often
		if (length(refracted) <= 0.5f) {
			bsdfWo = normalize(2.0f * dot(exitWi, h) * h - exitWi);
			itOri += exitNormalGeometrical * 0.0005f;
			continue;
		}
		// (TODO REVIEW)

		// (TODO REVIEW)
		// partial fresnel reflection back into the medium. only total
		// internal reflection was handled before, so every sub-critical
		// exit silently dropped its reflected fraction F -- an
		// angle-dependent energy loss that rings along iso-exit-angle
		// contours and starves obliquely-lit regions. clamped so the
		// branch probability and the weight divide use the same F
		const f32 fresnelExit = (
			min(
				utilMicrofacetFresnelDielectric(
					abs(dot(exitWi, h)), etaT / etaI
				),
				1.0f - 1e-4f
			)
		);
		if (fnSampleUniform(state) < fresnelExit) {
			bsdfWo = normalize(2.0f * dot(exitWi, h) * h - exitWi);
			itOri += exitNormalGeometrical * 0.0005f;
			continue;
		}
		// (TODO REVIEW)

		const f32 dotNorWo = dot(exitFrame.nor, refracted);
		const f32 exitPdf = (
			openPbrTransmissionPdf(
				exitMaterial, exitFrame, exitWi, refracted,
				/*isInsideMedium=*/true, /*dispersedIor=*/etaI
			)
		);
		if (exitPdf <= 0.0f) {
			return false;
		}
		// (TODO REVIEW)
		// the branch above already charged F, and openPbrTransmissionEvaluateF
		// carries its own (1 - F); dividing by the transmit probability
		// cancels it so fresnel is counted exactly once
		irradianceThroughput *= (
			openPbrTransmissionEvaluateF(
				exitMaterial, exitFrame, exitWi, refracted,
				/*isInsideMedium=*/true, /*dispersedIor=*/etaI
			)
			* abs(dotNorWo) / (exitPdf * (1.0f - fresnelExit))
		);
		// (TODO REVIEW)
		NAN_CHECK3(
			irradianceThroughput,
			"NaN utilSubsurfaceWalk.irradianceThroughput after exit divide px(%d,%d)=%v3f"
		)
		if (length(irradianceThroughput) < 1e-4f) {
			return false;
		}

		// successful refraction-out: the next segment travels in air, on
		// the opposite side from exitNormalGeometrical's inward-pointing
		// convention -- nudge along -normal, or the next utilTraceRay
		// starts from just inside the mesh and immediately re-hits this
		// same surface at ~0 distance instead of actually escaping
		bsdfWo = refracted;
		itOri += exitNormalGeometrical * -0.0005f;
		outExitPdf = exitPdf;
		return true;
	}
	// exceeded the step cap (pathologically high albedo); treat as
	// absorbed rather than looping forever
	return false;
}

// -----------------------------------------------------------------------------
// -- walk function
// -----------------------------------------------------------------------------

// path-walk was terminated from extinction
public static const u32 skWalkCap = 0u;
// path-walk intersected a surface
public static const u32 skWalkSurface = 1u;
// path-walk missed surface geometry and intersected the environment map
public static const u32 skWalkEnvironment = 2u;
// transports the path from the current vertex to its next interaction:
// traces along bsdfWo (the travel direction -- the camera direction on the
// first segment, the sampled bsdf direction afterward), random-walks the
// vdb medium along the way (each scatter re-aims bsdfWo via the phase
// function and re-traces), and on reaching a surface loads its material and
// shading frames. itWi is only ever *written* here (surface convention,
// -bsdfWo); it is never read as a direction of travel
public func utilIrradianceWalk(
	inout vdb: UtilVdb,
	inout material: OpenPbrMaterial,
	inout itOri: f32v3,
	inout itFrame: ShadingFrame,
	inout itCoatFrame: ShadingFrame,
	inout itNormalGeometrical: f32v3,
	inout itWi: f32v3,
	inout bsdfWo: f32v3,
	inout irradianceThroughput: f32v3,
	inout irradianceAccumulator: f32v3,
	out phasePdf: f32,
	isInsideMedium: bool,
	sampledTransmission: bool,
	out mediumScatter: bool,
	out mediumNeeScatter: bool,
	inout state: u64,
	unf: GpuGlobalExtended,
	envHandles: EnvironmentMapHandles,
	envNeeActive: bool,
	out outHit: RayQueryResult,
	out outSegmentOrigin: f32v3,
) -> i32 {
	mediumScatter = false;
	// (TODO REVIEW)
	mediumNeeScatter = false;
	outHit = RayQueryResult(0.0f, 0u, 0u, f32v2(0.0f));
	outSegmentOrigin = itOri;
	// (TODO REVIEW)


	// -- trace the segment along the current travel direction
	RayQueryResult rq = (
		utilTraceRay(
			/*origin=*/ itOri,
			/*dir=*/ bsdfWo,
			/*maxDist=*/999999.0f,
			/*isShadowRay=*/false,
			/*testAlphaCutoff=*/true,
			state
		)
	);

	// -- Beer-Lambert attenuation through transmissive medium (moved here
	// from utilIrradiancePropagate: this walk owns the segment length); a
	// thin-walled shell has no real interior volume to traverse, so this
	// only applies to solid dielectrics (khr_materials_volume with real
	// thickness). vdb scatter inside a dielectric interior is not handled:
	// attenuation uses the full straight segment
	// openpbr transmission_scatter: a scattering interior runs a real walk
	// below instead, which owns absorption too
	// (TODO REVIEW)
	// depth is the length scale both coefficients are expressed in, so
	// without it there is no medium to scatter in
	const bool interiorScatters = (
		sampledTransmission
		&& isInsideMedium
		&& material.geometryThinWalled <= 0.0f
		&& material.transmissionDepth > 0.0f
		&& any((material.transmissionScatter > f32v3(0.0f)))
	);
	// (TODO REVIEW)
	if (
		sampledTransmission
		&& isInsideMedium
		// (TODO REVIEW)
		&& !interiorScatters
		// (TODO REVIEW)
		&& material.geometryThinWalled <= 0.0f
		&& material.transmissionDepth > 0.0f
		&& rq.dist > 0.0f
	) {
		/*
			Beer-Lambert law (spec 3.7.4):\\
			\mu_t = -\frac{\ln T}{\lambda},
			\quad T = transmissionColor,
			\quad \lambda = transmissionDepth\\
			attenuation = \exp(-\mu_t \cdot d) = T^{d / \lambda}
		*/
		const f32v3 mu = (
			-log(max(material.transmissionColor, f32v3(1e-6f)))
			/ material.transmissionDepth
		);
		irradianceThroughput *= exp(-mu * rq.dist);
		// transmissionColor components near 1.0 make mu near 0, and dist can
		// be large for a long chord through the medium -- exp(-mu*dist) is
		// always finite on its own, but the same Inf-then-times-zero
		// overflow risk as the bsdf divide applies once this multiplies
		// into an already-large throughput
		NAN_CHECK3(
			irradianceThroughput,
			"NaN irradianceThroughput beer-lambert px(%d,%d)=%v3f"
		)
	}

	// (TODO REVIEW)
	// -- volumetric walk through a scattering dielectric interior. shares
	// the subsurface walk's chromatic free-flight step, which already
	// carries the hero-channel mis weights for both the scatter and the
	// exit branch. no nee from these vertices: inside a closed mesh every
	// shadow ray is occluded by the far wall, so light leaves at the
	// boundary through the exit interface's own nee
	if (interiorScatters) {
		/*
			openpbr spec, transmission scatter:\\
			\mu_s = S / \lambda, \quad
			\mu_t = -\ln(T) / \lambda, \quad
			\mu_a = \mu_t - \mu_s
			\\
			\mathrm{if} \; \min(\mu_a) < 0: \quad
			\mu_a \leftarrow \mu_a - \min(\mu_a)
			\\
			scattering is carved out of the extinction transmission_color
			already implies, not added on top
		*/
		const f32v3 sigmaScattering = (
			material.transmissionScatter / material.transmissionDepth
		);
		const f32v3 sigmaExtinction = (
			-log(max(material.transmissionColor, f32v3(1e-6f)))
			/ material.transmissionDepth
		);
		// (TODO REVIEW)
		// S can outrun the extinction transmission_color implies. openpbr
		// shifts the negative remainder by enough *gray* to clear zero, so
		// the hue survives; a per-channel clamp would tint the medium
		const f32v3 sigmaAbsorptionRaw = sigmaExtinction - sigmaScattering;
		const f32 sigmaAbsorptionMin = (
			min(
				min(sigmaAbsorptionRaw.r, sigmaAbsorptionRaw.g),
				sigmaAbsorptionRaw.b
			)
		);
		const f32v3 sigmaAbsorption = (
			sigmaAbsorptionMin < 0.0f
				? sigmaAbsorptionRaw - f32v3(sigmaAbsorptionMin)
				: sigmaAbsorptionRaw
		);
		// (TODO REVIEW)
		for (i32 s = 0; s < 64; ++s) {
			if (rq.dist <= 0.0f) {
				break;
			}
			f32 outDistance;
			f32v3 outWo;
			f32v3 outWeight;
			const u32 outcome = (
				openPbrSubsurfaceWalkStep(
					sigmaAbsorption, sigmaScattering, rq.dist, bsdfWo,
					material.transmissionScatterAnisotropy,
					// (TODO REVIEW)
					// no entry normal tracked here, so guiding stays off
					irradianceThroughput, f32v3(0.0f),
					// (TODO REVIEW)
					outDistance, outWo, outWeight, state
				)
			);
			irradianceThroughput *= outWeight;
			NAN_CHECK3(
				irradianceThroughput,
				"NaN irradianceThroughput transmission scatter px(%d,%d)=%v3f"
			)
			if (outcome != OPENPBR_SUBSURFACE_WALK_SCATTER) {
				break;
			}
			// phase-sampled, so the entry must not balance-weight an
			// env escape against a bsdf pdf
			mediumScatter = true;
			itOri += bsdfWo * outDistance;
			bsdfWo = outWo;
			if (length(irradianceThroughput) < 1e-4f) {
				return skWalkCap;
			}
			rq = utilTraceRay(itOri, bsdfWo, 999999.0f, false, true, state);
		}
	}
	// (TODO REVIEW)

	// -- random walk through the vdb medium and/or homogeneous fog along
	// the segment; vdb and fog are independent extinction fields occupying
	// the same space, so each step draws both candidate collision
	// distances and takes the nearer -- the poisson-superposition identity
	// (min of two independent free-flight draws = free-flight draw of the
	// combined field), not a probability-weighted lobe pick like the
	// phase-mixture sampling below
	phasePdf = 0.0f;
	if (vdb.isValid || unf.fogThickness > 0.0f) {
		const UtilVdbPhaseParams phaseParams = (
			utilVdbPhaseParamsForMedium(
				unf.vdbDropletDiameter, unf.vdbPhaseAnisotropyOverride
			)
		);
		for (i32 s = 0; s < 64; ++s) {
			const f32 distMax = rq.dist > 0.0f ? rq.dist : 999999.0f;
			// vdbCollisionWeight: the chromatic weighted-delta-tracking
			// correction utilVdbDeltaTrack accumulates (see its own header
			// comment) -- only meaningful when tScatterVdb >= 0, folded into
			// the throughput below alongside albedo
			f32v3 vdbCollisionWeight = f32v3(1.0f);
			const f32 tScatterVdb = (
				vdb.isValid
				? utilVdbDeltaTrack(
					vdb, itOri, bsdfWo, distMax,
					unf.vdbSigmaAbsorption, unf.vdbSigmaScattering, state,
					vdbCollisionWeight
				)
				: -1.0f
			);
			const f32 tScatterFogRaw = (
				unf.fogThickness > 0.0f
				? utilFogSampleDistance(unf.fogThickness, state)
				: -1.0f
			);
			const f32 tScatterFog = (
				tScatterFogRaw >= 0.0f
				&& tScatterFogRaw < min(distMax, unf.fogDistanceMax)
				? tScatterFogRaw
				: -1.0f
			);
			const bool fogWins = (
				tScatterFog >= 0.0f
				&& (tScatterVdb < 0.0f || tScatterFog <= tScatterVdb)
			);
			const f32 tScatter = fogWins ? tScatterFog : tScatterVdb;
			if (tScatter < 0.0f) {
				break;
			}
			mediumScatter = true;
			// (TODO REVIEW)
			// this is the vertex the nee block below samples from
			mediumNeeScatter = true;
			// (TODO REVIEW)
			itOri += bsdfWo * tScatter;

			// -- blackbody emission (fire): sampled from a second,
			// optional temperature grid at this same collision point,
			// added using the throughput as it stands entering this
			// vertex -- emission getting out of the medium here doesn't
			// depend on whether/how the path continues past x, so it must
			// not be weighted by this vertex's own albedo (added below),
			// the same convention surface emission uses (throughput prior
			// to sampling the next bounce). see util-blackbody.glsl
			if (!fogWins && unf.vdbTemperatureGrid != 0u) {
				UtilVdb vdbTemperature = (
					utilVdbBind(unf.vdbTemperatureGrid)
				);
				vdbTemperature.worldToLocal = vdb.worldToLocal;
				const f32 temperatureKelvin = (
					utilVdbDensity(vdbTemperature, itOri) * unf.vdbTemperatureScale
				);
				if (temperatureKelvin > 0.0f) {
					irradianceAccumulator += (
						irradianceThroughput
						* unf.vdbEmissionScale
						* utilBlackbodyRadiance(temperatureKelvin)
					);
				}
				// utilVdbBind above repointed the shared global bda
				// buffer at the temperature grid; restore it before vdb
				// (density) is touched again
				utilVdbRebind(vdb);
			}

			irradianceThroughput *= (
				fogWins
				? unf.fogAlbedo
				: (
					vdbCollisionWeight
					* utilVdbAlbedo(unf.vdbSigmaAbsorption, unf.vdbSigmaScattering)
				)
			);

			// -- next event estimation from the medium vertex, mirroring
			// the surface nee block above with a phase function in place
			// of the bsdf f. same combined pick over the light list, so the
			// environment and the emissive meshes share one shadow ray here
			// too; an empty list takes no draws and leaves the pre-P4 stream
			const LightPick mediumPick = utilLightPick(unf, envNeeActive, state);
			if (mediumPick.envPicked) {
				const f32v2 xi = (
					mediumPick.envFromList
						? f32v2(mediumPick.xi.x, mediumPick.xi.y)
						: f32v2(fnSampleUniform(state), fnSampleUniform(state))
				);
				f32 envPdf;
				const f32v3 envDir = (
					environmentMapImportanceSample(
						envHandles, xi, unf.envRotation, envPdf
					)
				);
				// the light technique's real density is pick * map
				envPdf *= mediumPick.envPickPdf;
				if (envPdf > 0.0f) {
					const f32 phaseAtEnv = (
						fogWins
						? skUtilFogPhaseIsotropic
						: utilVdbPhaseFog(dot(bsdfWo, envDir), phaseParams)
					);
					const RayQueryResult visibilityRq = (
						utilTraceRay(
							itOri,
							envDir,
							/*maxDist=*/999999.0f,
							/*isShadowRay=*/true,
							/*testAlphaCutoff=*/true,
							state
						)
					);
					if (visibilityRq.dist <= 0.0f) {
						const f32v3 transmittance = (
							utilVdbFogTransmittance(
								vdb, itOri, envDir,
								unf.vdbSigmaAbsorption, unf.vdbSigmaScattering,
								unf.fogThickness,
								unf.fogDistanceMax,
								// (TODO REVIEW)
								/*travelDistance=*/999999.0f,
								// (TODO REVIEW)
								state
							)
						);
						if (
							max(transmittance.x, max(transmittance.y, transmittance.z))
							> 0.0f
						) {
							const f32v3 envRadiance = (
								environmentMapRadiance(
									envHandles, envDir, unf.envRotation
								)
								* unf.envIntensity
							);
							irradianceAccumulator += (
								irradianceThroughput
								* phaseAtEnv * envRadiance * transmittance
								* misBalanceWeight(envPdf, phaseAtEnv)
								/ envPdf
							);
						}
					}
				}
			}

			// (TODO REVIEW)
			// -- the mesh branch of the same pick. no surface here, so the
			// shadow ray starts at the scatter point itself: es.direction and
			// es.distance are both measured from it, and section 3.7's
			// offset-origin trap cannot apply
			if (mediumPick.meshPicked) {
				const EmitterSample es = (
					utilEmitterSampleFromPick(
						(GpuEmitterLight *)(unf.emitterLights),
						mediumPick.pick,
						(GpuResolveModelIndirect *)(gUtilRaytraceModels),
						itOri,
						mediumPick.xi
					)
				);
				if (es.valid) {
					const f32 phaseAtEmitter = (
						fogWins
						? skUtilFogPhaseIsotropic
						: utilVdbPhaseFog(dot(bsdfWo, es.direction), phaseParams)
					);
					const RayQueryResult emitterRq = (
						utilTraceRay(
							itOri,
							es.direction,
							/*maxDist=*/ es.distance * (1.0f - 1e-4f),
							/*isShadowRay=*/true,
							/*testAlphaCutoff=*/true,
							state
						)
					);
					if (emitterRq.dist <= 0.0f) {
						const f32v3 transmittance = (
							utilVdbFogTransmittance(
								vdb, itOri, es.direction,
								unf.vdbSigmaAbsorption, unf.vdbSigmaScattering,
								unf.fogThickness,
								unf.fogDistanceMax,
								// (TODO REVIEW)
								// the connection stops at the emitter
								/*travelDistance=*/ es.distance,
								// (TODO REVIEW)
								state
							)
						);
						if (
							max(transmittance.x, max(transmittance.y, transmittance.z))
							> 0.0f
						) {
							const f32v3 emitterRadiance = (
								utilEmitterSampleRadiance(
									(GpuEmitterLight *)(unf.emitterLights), es
								)
							);
							irradianceAccumulator += (
								irradianceThroughput
								* phaseAtEmitter * emitterRadiance * transmittance
								* misBalanceWeight(es.pdf, phaseAtEmitter)
								/ es.pdf
							);
						}
					}
				}
			}
			// (TODO REVIEW)


			// -- re-aim via the phase function, record its pdf
			const f32v3 incomingDir = bsdfWo;
			if (fogWins) {
				bsdfWo = fnSampleUniformSphere(fnSampleUniform2(state));
				phasePdf = skUtilFogPhaseIsotropic;
			} else {
				bsdfWo = utilVdbPhaseSampleMixture(phaseParams, bsdfWo, state);
				phasePdf = utilVdbPhaseFog(dot(incomingDir, bsdfWo), phaseParams);
			}
			rq = (
				utilTraceRay(
					/*origin=*/ itOri,
					/*dir=*/ bsdfWo,
					/*maxDist=*/999999.0f,
					/*isShadowRay=*/false,
					/*testAlphaCutoff=*/true,
					state
				)
			);
		}
		if (mediumScatter && length(irradianceThroughput) < 1e-4f) {
			return skWalkCap;
		}
	}

	if (rq.dist <= 0.0f) {
		// missed surface; the entry terminates with the env contribution
		return skWalkEnvironment;
	}

	// (TODO REVIEW)
	// itOri has been advanced by every medium scatter along the way, so this
	// is the origin of the final segment -- the vertex the light-sampling
	// technique would have connected from, which is what its pdf is relative to
	outHit = rq;
	outSegmentOrigin = itOri;
	// (TODO REVIEW)

	// -- update material and shading frame
	utilLoadSurfaceHit(
		rq, itOri, bsdfWo,
		material, itFrame, itCoatFrame, itNormalGeometrical, itWi
	);

	return skWalkSurface;
}

// -----------------------------------------------------------------------------
// -- irradiance propagation entry point
// -----------------------------------------------------------------------------

public func utilIrradiancePropagationEntry(
	tables: MaterialTableHandles,
	inout vdb: UtilVdb,
	origin: f32v3,
	worldDir: f32v3,
	propagationDepth: u32,
	inout state: u64,
	unf: GpuGlobalExtended,
) -> f32v4 {
	// glsl parameters are mutable copies; slang's are not
	var origin = origin;
	f32v3 irradianceThroughput = f32v3(1.0);
	f32v3 irradianceAccumulator = f32v3(0.0);

	// -- per-path state; the walk's surface setup fills material/frames/wi
	// before the first surface vertex ever reads them
	bool isInsideMedium = false;
	i32 pathSpectralChannel = -1;
	OpenPbrMaterial material;
	ShadingFrame frame;
	ShadingFrame coatFrame;
	f32v3 normalGeometrical = f32v3(0.0f);
	f32v3 wi = f32v3(0.0f);
	f32v3 bsdfWo = worldDir;
	f32 bsdfPdf = 0.0f;
	bool sampledTransmission = false;
	// (TODO REVIEW)
	// emissive-mesh mis path state. the camera vertex ran no light sampling,
	// so a primary ray landing straight on an emitter takes its full radiance
	bool previousVertexSampledLights = false;
	bool previousBounceWasDelta = false;
	// the environment's share of the previous vertex's light pick; 0 until a
	// vertex has actually run env nee, so a primary ray straight into the env
	// map keeps its full radiance
	f32 previousEnvPickPdf = 0.0f;
	// (TODO REVIEW)

	// -- env handles for the escape branch (mirrors utilIrradiance's fetch)
	const EnvironmentMapHandles envHandles = (
		unf.envMode == ENV_MODE_HDRMAP
			? ((EnvironmentMapHandles *)(unf.envMap))[0]
			: EnvironmentMapHandles(0u, 0u, 0u, 0u)
	);
	const bool envNeeActive = (
		unf.envMode == ENV_MODE_HDRMAP && unf.envNeeEnabled != 0u
	);

	i32 hit = 0;
	for (i32 i = 0; i < propagationDepth; ++i) {
		// (TODO REVIEW)
		// past envMapOnlyDepth surfaces, resolve the bsdf-sampled
		// continuation straight against the environment -- no AS trace,
		// so e.g. a mirror sphere reflects the env map in place of a
		// neighboring object instead of that ray either cross-lighting
		// the scene or (if just discarded) leaving a black hole where a
		// real trace would have hit geometry
		// (TODO REVIEW)
		const bool envMapOnlyReached = (
			unf.envMapOnlyDepth != 0u && i >= i32(unf.envMapOnlyDepth)
		);
		// depth >= 2 is only ever the drag preview, which is full-screen every
		// moved frame (the tile budget skips View), so a rough bounce resolves
		// one segment early instead of paying a second trace -- it reads
		// slightly brighter than the final render, since the env stands in for
		// whatever geometry that ray would have hit. gated on the depth rather
		// than frameMode: unit-test harnesses leave frameMode at View
		const bool viewRoughEarlyOut = (
			unf.envMapOnlyDepth >= 2u
			&& i > 0
			&& !sampledTransmission
			&& min(
				material.specularRoughness,
				material.coatWeight > 0.0f ? material.coatRoughness : 1.0f
			) > skViewPreviewSmoothRoughness
		);
		if (envMapOnlyReached || viewRoughEarlyOut) {
			// (TODO REVIEW)
			const f32v3 radiance = (
				sampleEnvironment(
					/*dir=*/ bsdfWo,
					/*intensity=*/ unf.envIntensity,
					/*envMode=*/ unf.envMode,
					/*envHandles=*/ envHandles,
					/*envRotation=*/ unf.envRotation
				)
			);
			if (any(radiance != f32v3(0.0f))) {
				f32 weightB = 1.0f;
				// (TODO REVIEW)
				// previousEnvPickPdf, not envNeeActive: the competing density
				// is the *pick* times the map's, and it is 0 at a vertex that
				// ran no env nee at all -- misBalanceWeight(bsdfPdf, 0) is
				// then 1, which is what an unreachable-by-nee path needs
				if (previousEnvPickPdf > 0.0f && !previousBounceWasDelta) {
					const f32 envPdf = (
						environmentMapPdf(envHandles, bsdfWo, unf.envRotation)
					);
					weightB = (
						misBalanceWeight(bsdfPdf, previousEnvPickPdf * envPdf)
					);
				}
				// (TODO REVIEW)
				irradianceAccumulator += radiance * irradianceThroughput * weightB;
				hit = 1;
			} else if (any((irradianceAccumulator > f32v3(0.0f)))) {
				hit = 1;
			}
			break;
		}

		bool mediumScatter;
		// (TODO REVIEW)
		bool mediumNeeScatter;
		// (TODO REVIEW)
		f32 phasePdf;
		// (TODO REVIEW)
		RayQueryResult walkHit;
		f32v3 walkSegmentOrigin;
		// (TODO REVIEW)
		const i32 walkEnum = (
			utilIrradianceWalk(
				vdb,
				material,
				origin,
				frame,
				coatFrame,
				normalGeometrical,
				wi,
				bsdfWo,
				irradianceThroughput,
				irradianceAccumulator,
				phasePdf,
				isInsideMedium,
				sampledTransmission,
				mediumScatter,
				// (TODO REVIEW)
				mediumNeeScatter,
				// (TODO REVIEW)
				state,
				unf,
				envHandles,
				envNeeActive,
				walkHit,
				walkSegmentOrigin
			)
		);
		// TODO(spp validity): a medium scatter alone should count as a
		// valid sample; skWalkCap paths currently only report hit when
		// something was already accumulated
		if (walkEnum == skWalkCap) {
			if (any((irradianceAccumulator > f32v3(0.0f)))) {
				hit = 1;
			}
			break;
		}
		if (walkEnum == skWalkEnvironment) {
			// -- environment contribution; a miss always terminates the
			// path (moved here from utilIrradiance's bounce-miss block)
			const f32v3 radiance = (
				sampleEnvironment(
					/*dir=*/ bsdfWo,
					/*intensity=*/ unf.envIntensity,
					/*envMode=*/ unf.envMode,
					/*envHandles=*/ envHandles,
					/*envRotation=*/ unf.envRotation
				)
			);
			if (any(radiance != f32v3(0.0f))) {
				// mis-weight heuristics for bsdf-sampled contribution
				// against nee's estimate of the same env direction
				f32 weightB = 1.0f;
				// (TODO REVIEW)
				// a vdb/fog scatter runs its own light pick, so that escape
				// balances against the env entry's share of *that* pick -- not 1,
				// which only holds when the env is alone in the list. off a
				// surface the competing density is the previous vertex's own
				// share; 0, hence weight 1, wherever no env nee ran, the bounce
				// was delta, or the scatter was a transmission-interior one that
				// sampled no lights at all
				const f32 escapeEnvPickPdf = (
					mediumNeeScatter
						? utilLightEnvPickPdf(unf, envNeeActive)
						: (
							(mediumScatter || previousBounceWasDelta)
								? 0.0f : previousEnvPickPdf
						)
				);
				if (escapeEnvPickPdf > 0.0f) {
					const f32 envPdf = (
						escapeEnvPickPdf
						* environmentMapPdf(envHandles, bsdfWo, unf.envRotation)
					);
					if (mediumScatter) {
						// heuristic for phase-sampled escape
						weightB = misBalanceWeight(phasePdf, envPdf);
					} else if (i > 0) {
						// heuristic for bsdf-sampled escape
						weightB = misBalanceWeight(bsdfPdf, envPdf);
					}
				}
				// (TODO REVIEW)
				// if this is the first walk segment, then multiply the
				// irradiance by user selected background intensity, but
				// only if no medium was sampled along the way
				const f32 backgroundIntensity = (
					(i == 0 && !mediumScatter) ? unf.envBackgroundIntensity : 1.0f
				);
				irradianceAccumulator += (
					radiance * irradianceThroughput * weightB * backgroundIntensity
				);
				hit = 1;
			} else if (any((irradianceAccumulator > f32v3(0.0f)))) {
				// a black environment still ends the path; whatever emission
				// was accumulated on the way here is the path's contribution
				hit = 1;
			}
			break;
		}

		// -- surface vertex
		// (TODO REVIEW)
		// a medium scatter re-aimed the ray with the phase function, so that
		// is the density competing with light sampling for this segment
		const f32 previousSamplePdf = mediumScatter ? phasePdf : bsdfPdf;
		// a medium scatter is the vertex this segment actually left from, so its
		// light-sampling state replaces whatever the last surface vertex
		// recorded. a vdb/fog scatter runs the same combined pick a surface does,
		// mesh entries included -- reporting false there double counts an
		// emissive triangle this phase-sampled ray goes on to hit
		if (mediumScatter) {
			previousVertexSampledLights = (
				mediumNeeScatter && unf.emitterLightCount > 0u
			);
			previousBounceWasDelta = false;
		} else if (i == 0) {
			// the camera vertex itself ran no light sampling. this cannot be
			// folded into the call below as `i > 0 && ...`: a medium scatter on
			// the very first segment *is* a preceding light-sampling event, and
			// suppressing it double counts every emitter that scatter then hits
			previousVertexSampledLights = false;
		}
		bool vertexSampledLights;
		f32 vertexEnvPickPdf;
		bool sampledDelta;
		// (TODO REVIEW)
		const i32 returnEnum = (
			utilIrradiancePropagate(
				tables,
				vdb,
				material,
				origin,
				frame,
				coatFrame,
				normalGeometrical,
				wi,
				bsdfWo,
				bsdfPdf,
				irradianceThroughput,
				irradianceAccumulator,
				state,
				i,
				isInsideMedium,
				sampledTransmission,
				pathSpectralChannel,
				unf,
				envHandles,
				envNeeActive,
				walkHit.modelDrawIndex,
				walkHit.primitiveIndex,
				walkHit.dist,
				walkSegmentOrigin,
				previousSamplePdf,
				previousBounceWasDelta,
				previousVertexSampledLights,
				vertexSampledLights,
				vertexEnvPickPdf,
				sampledDelta
			)
		);
		// (TODO REVIEW)
		previousVertexSampledLights = vertexSampledLights;
		previousEnvPickPdf = vertexEnvPickPdf;
		previousBounceWasDelta = sampledDelta;
		// (TODO REVIEW)

		if (returnEnum == skHitIndirect) {
			hit = 1;
		}

		if (returnEnum == skHitTerminate) {
			// terminate still permits using whatever was already
			// accumulated (e.g. emission seen before an invalid bsdf
			// sample cut the path short); only a bare miss reports no hit
			if (any((irradianceAccumulator > f32v3(0.0f)))) {
				hit = 1;
			}
			break;
		}
	}

	// debug for NaN hunts
	if (any(isnan(irradianceAccumulator)) && hit > 0) {
		return f32v4(9999.0f, 0.0f, 0.0f, 1.0f);
	}

	return f32v4(irradianceAccumulator, f32(hit));
}

