// util to trace rays in the scene using ray queries

#ifndef UTIL_RAYTRACE_GLSL
#define UTIL_RAYTRACE_GLSL

#include "util-mesh-raytrace.glsl"
#include "util-shading-frame.glsl"

#include "util-vdb.glsl"
#include "util-blackbody.glsl"
#include "util-material-openpbr.glsl"
#include "util-material-openpbr-load.glsl"

// -----------------------------------------------------------------------------
// -- ray-query utility
// -----------------------------------------------------------------------------

struct RayQueryResult {
	float dist;
	uint modelDrawIndex;
	uint primitiveIndex;
	f32v2 barycentric;
};

// (TODO REVIEW)
// openpbr geometry_opacity resolved to a coverage fraction: the alpha in
// M_pbr = mix(S_ambient_medium, M_surface, alpha). gltf MASK quantizes it
// to 0/1 (spec: opaque when alpha >= cutoff), BLEND keeps it fractional,
// OPAQUE arrives already forced to 1 by mor's loader
float utilAlphaCoverage(const OpenPbrMaterial mat) {
	if (mat.alphaMode == MOR_ALPHA_MODE_MASK) {
		return mat.geometryOpacity >= mat.alphaCutoff ? 1.0f : 0.0f;
	}
	return mat.geometryOpacity;
}

// stochasticCoverage: BLEND surfaces pass the ray straight through with
// probability 1-alpha instead of thresholding. unbiased, and because it
// resolves inside traversal the pass-through costs no bounce and no re-trace
RayQueryResult utilTraceRayImpl(
	const vec3 origin,
	const vec3 dir,
	const float maxDist,
	const bool isShadowRay,
	const bool testAlphaCutoff,
	const bool stochasticCoverage,
	inout u64 state
) {
	rayQueryEXT rq;
	uint rayFlags = 0;
	if (!testAlphaCutoff) {
		rayFlags |= gl_RayFlagsOpaqueEXT;
	}
	if (isShadowRay) {
		rayFlags |= gl_RayFlagsTerminateOnFirstHitEXT;
	}
	rayQueryInitializeEXT(
		rq,
		/*tlas*/ vkofTlas,
		/*flags*/ rayFlags,
		/*cullMask*/ 0xFF,
		/*origin*/ origin,
		/*minT*/ 0.00001f,
		/*dir*/ dir,
		/*maxT*/ maxDist
	);

	while (rayQueryProceedEXT(rq)) {
		if (
			rayQueryGetIntersectionTypeEXT(rq, false)
			!= gl_RayQueryCandidateIntersectionTriangleEXT
		) {
			// procedural AABB candidates are unsupported; leave unconfirmed
			continue;
		}

		UtilMeshAttributeDataFromIndices rqData = (
			utilMeshAttributeDataFromIndices(
				rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false),
				rayQueryGetIntersectionPrimitiveIndexEXT(rq, false),
				rayQueryGetIntersectionBarycentricsEXT(rq, false),
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
		const float coverage = utilAlphaCoverage(mat);
		const bool covered = (
			stochasticCoverage
			? fnSampleUniform(state) < coverage
			: coverage >= 0.5f
		);
		if (covered) {
			rayQueryConfirmIntersectionEXT(rq);
			if (isShadowRay) {
				rayQueryTerminateEXT(rq);
			}
		}
	}

	// read back the committed intersection; candidates can arrive in any
	// order, so the last candidate examined is not necessarily the closest
	if (
		rayQueryGetIntersectionTypeEXT(rq, true)
		== gl_RayQueryCommittedIntersectionTriangleEXT
	) {
		RayQueryResult result;
		result.dist = rayQueryGetIntersectionTEXT(rq, true);
		result.modelDrawIndex = (
			rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true)
		);
		result.primitiveIndex = (
			rayQueryGetIntersectionPrimitiveIndexEXT(rq, true)
		);
		result.barycentric = (
			rayQueryGetIntersectionBarycentricsEXT(rq, true)
		);
		return result;
	}

	return RayQueryResult(0.0f, 0u, 0u, f32v2(0.0f));
}

// path-tracing entry: BLEND coverage resolves stochastically against the
// path's rng stream
RayQueryResult utilTraceRay(
	const vec3 origin,
	const vec3 dir,
	const float maxDist,
	const bool isShadowRay,
	const bool testAlphaCutoff,
	inout u64 state
) {
	return utilTraceRayImpl(
		origin, dir, maxDist, isShadowRay, testAlphaCutoff,
		/*stochasticCoverage=*/true, state
	);
}

// aov/picking/debug entry: no rng stream to spend, and a stochastic hit
// would make picking and single-sample probes flicker frame to frame, so
// BLEND collapses to a half-coverage threshold
RayQueryResult utilTraceRay(
	const vec3 origin,
	const vec3 dir,
	const float maxDist,
	const bool isShadowRay,
	const bool testAlphaCutoff
) {
	u64 unusedState = 0ul;
	return utilTraceRayImpl(
		origin, dir, maxDist, isShadowRay, testAlphaCutoff,
		/*stochasticCoverage=*/false, unusedState
	);
}
// (TODO REVIEW)

// -----------------------------------------------------------------------------
// -- irradiance propagation
// -----------------------------------------------------------------------------

// no contributions were found. It's still okay to continue bouncing as well
// it's okay to use the accumulated irradiance. If there's no accumulated
// irradiance and the path is terminated, then no overall contribution
#define skHitNone 0

// an indirect contribution was found. The path can continue bouncing, and
// on path termination the accumulated irradiance can be used.
#define skHitIndirect 1

// the path should be terminated. Any contribution can be used however
// if there is none accumulated, then no overall contribution.
#define skHitTerminate 2

struct Ray {
	f32v3 origin;
	f32v3 dir;
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

int utilIrradiance(
	const MaterialTableHandles tables,
	inout UtilVdb vdb,
	const OpenPbrMaterial material,
	inout f32v3 itOri,
	inout ShadingFrame itFrame,
	inout ShadingFrame itCoatFrame,
	inout f32v3 itNormalGeometrical,
	inout f32v3 itWi,
	inout f32v3 irradianceThroughput,
	inout f32v3 irradianceAccumulator,
	out f32v3 bsdfWo,
	// pdf of the sampled bsdfWo; the entry keeps it as path state so the
	// env-escape mis reweight (previously the bounce-miss block here) can
	// balance against nee when the next walk segment reaches the env map
	out f32 bsdfPdf,
	inout u64 state,
	out bool sampledTransmission,
	const int iteration,
	const bool isInsideMedium,
	// -1 = no hero wavelength locked yet (path hasn't sampled a dispersive
	// transmission event); once locked (0/1/2 = R/G/B), stays locked for
	// the rest of the path -- see the dispersion resolution block below
	inout int pathSpectralChannel,
	// path-invariant; caller loads these once instead of every bounce
	// re-dereferencing the same buffer_reference data
	const GpuGlobalExtended unf,
	const EnvironmentMapHandles envHandles,
	const bool envNeeActive
) {
	int returnEnum = skHitNone;

	// -- surface emission at this vertex, weighted by the throughput carried
	// into it; plain bsdf path tracing, every vertex accumulates its own
	// emission
	const f32v3 emission = material.emissionColor * material.emissionLuminance;
	if (any(greaterThan(emission, f32v3(0.0f)))) {
		irradianceAccumulator += emission * irradianceThroughput;
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
	const int provisionalChannel = (
		dispersionAlreadyLocked
			? pathSpectralChannel
			: min(int(fnSampleUniform(state) * 3.0f), 2)
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

	// -- next event estimation: importance-samples the env map directly
	// instead of relying on a bsdf-sampled bounce to hit it by chance,
	// mis-combined with the bsdf-sampled technique below (see the bounce-
	// miss block's weightB). uses the incoming irradianceThroughput --
	// before this bounce's own bsdf f*cos/pdf update -- since this is a
	// same-vertex direct-lighting estimate, not part of the indirect
	// continuation. dispersedIor is already resolved above and reused as
	// -is; nee never calls openPbrSampleWo, so it can't trigger the
	// dispersion provisional -> locked commit below
	if (envNeeActive) {
		const f32v2 xi = f32v2(fnSampleUniform(state), fnSampleUniform(state));
		f32 envPdf;
		const f32v3 envDir = (
			environmentMapImportanceSample(
				envHandles, xi, unf.envRotation, envPdf
			)
		);
		if (envPdf > 0.0f) {
			const f32v3 f = (
				openPbrEvaluateF(
					tables, itFrame, itCoatFrame, itWi, envDir, material,
					isInsideMedium, dispersedIor
				)
			);
			// cheap early-out before tracing a shadow ray
			if (any(greaterThan(f, f32v3(0.0f)))) {
				const f32 bsdfPdfAtEnvDir = (
					openPbrEvaluatePdf(
						tables, material, itFrame, itCoatFrame, itWi, envDir,
						isInsideMedium, dispersedIor
					)
				);
				const f32 weightL = misBalanceWeight(envPdf, bsdfPdfAtEnvDir);
				// offset toward whichever side of the surface envDir
				// actually points at (unlike the bsdf-sampled bounce below,
				// there's no sampledTransmission flag for an nee direction
				// drawn straight from the env map)
				const f32v3 shadowOrigin = (
					itOri
					+ itNormalGeometrical * (
						dot(itNormalGeometrical, envDir) >= 0.0f
							? 0.0005f : -0.0005f
					)
				);
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
				if (visibilityRq.dist <= 0.0f) {
					const f32v3 transmittance = (
						utilVdbFogTransmittance(
							vdb, shadowOrigin, envDir,
							unf.vdbSigmaAbsorption, unf.vdbSigmaScattering,
							unf.fogThickness,
							unf.fogDistanceMax, state
						)
					);
					const f32v3 envRadiance = (
						environmentMapRadiance(envHandles, envDir, unf.envRotation)
						* unf.envIntensity
					);
					irradianceAccumulator += (
						irradianceThroughput * f
						* abs(dot(itFrame.nor, envDir)) * envRadiance
						* weightL / envPdf
						* transmittance
					);
				}
			}
		}
	}

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

int utilIrradiancePropagate(
	const MaterialTableHandles tables,
	inout UtilVdb vdb,
	inout OpenPbrMaterial material,
	inout f32v3 itOri,
	inout ShadingFrame itFrame,
	inout ShadingFrame itCoatFrame,
	inout f32v3 itNormalGeometrical,
	inout f32v3 itWi,
	inout f32v3 bsdfWo,
	out f32 bsdfPdf,
	inout f32v3 irradianceThroughput,
	inout f32v3 irradianceAccumulator,
	inout u64 state,
	const int iteration,
	inout bool isInsideMedium,
	// path state: the next walk segment applies beer-lambert attenuation
	// when this vertex transmitted into a solid dielectric interior
	out bool sampledTransmission,
	inout int pathSpectralChannel,
	const GpuGlobalExtended unf,
	const EnvironmentMapHandles envHandles,
	const bool envNeeActive
) {
	// -- compute irradiance on surface
	int returnEnum = (
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
			envNeeActive
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
void utilLoadSurfaceHit(
	const RayQueryResult rq,
	inout f32v3 itOri,
	const f32v3 bsdfWo,
	out OpenPbrMaterial material,
	out ShadingFrame itFrame,
	out ShadingFrame itCoatFrame,
	out f32v3 itNormalGeometrical,
	out f32v3 itWi
) {
	UtilMeshAttributeDataFromIndices rqData = (
		utilMeshAttributeDataFromIndices(
			rq.modelDrawIndex,
			rq.primitiveIndex,
			rq.barycentric,
			/*unpackUvDerivatives=*/true
		)
	);
	const mat3 tbn = (
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
				tbn * geometryNormal, rqData.normalGeometrical
			),
			tbn * geometryTangent.xyz
		)
	);
	// coat's own shading normal + tangent (khr_materials_clearcoat has no
	// tangent of its own, so it rides the same identity fallback)
	itCoatFrame = (
		shadingFrameFromTangent(
			utilShadingNormalOrGeometric(
				tbn * geometryCoatNormal, rqData.normalGeometrical
			),
			tbn * geometryCoatTangent.xyz
		)
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
#define skSubsurfaceWalkMaxSteps 4096

// bounces before russian roulette starts culling the walk; a handful of
// free bounces up front, matching the outer path's own "iteration > 2"
// warmup in utilIrradiance
#define skSubsurfaceWalkRRWarmup 4

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
	out f32 outExitPdf
) {
	outExitPdf = 0.0f;
	f32v3 sigmaAbsorption;
	f32v3 sigmaScattering;
	openPbrSubsurfaceAlbedoToSigma(
		entryMaterial.subsurfaceColor,
		entryMaterial.subsurfaceRadius * entryMaterial.subsurfaceRadiusScale,
		entryMaterial.subsurfaceScatterAnisotropy,
		sigmaAbsorption, sigmaScattering
	);

	for (int step = 0; step < skSubsurfaceWalkMaxSteps; ++step) {
		const RayQueryResult rq = (
			utilTraceRay(itOri, bsdfWo, 999999.0f, false, true, state)
		);
		if (rq.dist <= 0.0f) {
			return false;
		}

		f32 outDistance;
		f32v3 outWo;
		f32v3 outWeight;
		const uint outcome = (
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
		if (length(refracted) <= 0.5f) {
			// total internal reflection: stays inside, walk continues
			// from this same boundary point back into the medium.
			// exitNormalGeometrical points inward (utilLoadSurfaceHit's
			// back-face flip, same convention utilIrradiance's own tail
			// nudge relies on), so continuing inside nudges along +normal
			bsdfWo = normalize(2.0f * dot(exitWi, h) * h - exitWi);
			itOri += exitNormalGeometrical * 0.0005f;
			continue;
		}

		// -- nee from the exit vertex: same shadow-ray/mis pattern as
		// utilIrradiance's own env nee block, narrowed to just this
		// dielectric interface -- the walk already stands in for the
		// diffuse/subsurface response, so evaluating the full material
		// stack here (coat/fuzz/diffuse/another subsurface entry) would
		// double count it
		if (envNeeActive) {
			const f32v2 xiNee = fnSampleUniform2(state);
			f32 envPdf;
			const f32v3 envDir = (
				environmentMapImportanceSample(
					envHandles, xiNee, unf.envRotation, envPdf
				)
			);
			if (envPdf > 0.0f) {
				const f32v3 f = (
					openPbrTransmissionEvaluateF(
						exitMaterial, exitFrame, exitWi, envDir,
						/*isInsideMedium=*/true,
						/*dispersedIor=*/etaI
					)
				);
				if (any(greaterThan(f, f32v3(0.0f)))) {
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
					const RayQueryResult visibilityRq = (
						utilTraceRay(
							shadowOrigin, envDir, /*maxDist=*/999999.0f,
							/*isShadowRay=*/true, /*testAlphaCutoff=*/true,
							state
						)
					);
					if (visibilityRq.dist <= 0.0f) {
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
#define skWalkCap 0

// path-walk intersected a surface
#define skWalkSurface 1

// path-walk missed surface geometry and intersected the environment map
#define skWalkEnvironment 2

// transports the path from the current vertex to its next interaction:
// traces along bsdfWo (the travel direction -- the camera direction on the
// first segment, the sampled bsdf direction afterward), random-walks the
// vdb medium along the way (each scatter re-aims bsdfWo via the phase
// function and re-traces), and on reaching a surface loads its material and
// shading frames. itWi is only ever *written* here (surface convention,
// -bsdfWo); it is never read as a direction of travel
int utilIrradianceWalk(
	inout UtilVdb vdb,
	inout OpenPbrMaterial material,
	inout f32v3 itOri,
	inout ShadingFrame itFrame,
	inout ShadingFrame itCoatFrame,
	inout f32v3 itNormalGeometrical,
	inout f32v3 itWi,
	inout f32v3 bsdfWo,
	inout f32v3 irradianceThroughput,
	inout f32v3 irradianceAccumulator,
	// phase pdf of the scatter's sampled direction
	// must be used as an mis weight in estimator heuristics
	out f32 phasePdf,
	const bool isInsideMedium,
	const bool sampledTransmission,
	// whether the vdb medium scattered on this segment; the entry needs it
	// for the env-escape mis weight (a phase-sampled escape direction has
	// no competing nee estimator, so it must not be balance-weighted)
	out bool mediumScatter,
	inout u64 state,
	const GpuGlobalExtended unf,
	const EnvironmentMapHandles envHandles,
	const bool envNeeActive
) {
	mediumScatter = false;

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
		&& any(greaterThan(material.transmissionScatter, f32v3(0.0f)))
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
			const uint outcome = (
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
			// of the bsdf f
			if (envNeeActive) {
				const f32v2 xi = (
					f32v2(fnSampleUniform(state), fnSampleUniform(state))
				);
				f32 envPdf;
				const f32v3 envDir = (
					environmentMapImportanceSample(
						envHandles, xi, unf.envRotation, envPdf
					)
				);
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
								unf.fogDistanceMax, state
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

f32v4 utilIrradiancePropagationEntry(
	const MaterialTableHandles tables,
	inout UtilVdb vdb,
	f32v3 origin,
	// travel direction of the camera ray (NOT surface-convention wi);
	// resolve.comp passes rayDir
	const f32v3 worldDir,
	const uint propagationDepth,
	inout u64 state,
	// caller (resolve.comp) already loaded this for its own use; every
	// bounce used to re-dereference the same buffer_reference data instead
	// of reusing it
	const GpuGlobalExtended unf
) {
	f32v3 irradianceThroughput = f32v3(1.0);
	f32v3 irradianceAccumulator = f32v3(0.0);

	// -- per-path state; the walk's surface setup fills material/frames/wi
	// before the first surface vertex ever reads them
	bool isInsideMedium = false;
	int pathSpectralChannel = -1;
	OpenPbrMaterial material;
	ShadingFrame frame;
	ShadingFrame coatFrame;
	f32v3 normalGeometrical = f32v3(0.0f);
	f32v3 wi = f32v3(0.0f);
	f32v3 bsdfWo = worldDir;
	f32 bsdfPdf = 0.0f;
	bool sampledTransmission = false;

	// -- env handles for the escape branch (mirrors utilIrradiance's fetch)
	const EnvironmentMapHandles envHandles = (
		unf.envMode == ENV_MODE_HDRMAP
			? EnvironmentMapHandlesBuffer(unf.envMap).data
			: EnvironmentMapHandles(0u, 0u, 0u, 0u)
	);
	const bool envNeeActive = (
		unf.envMode == ENV_MODE_HDRMAP && unf.envNeeEnabled != 0u
	);

	int hit = 0;
	for (int i = 0; i < propagationDepth; ++i) {
		// diagnostic: past the first surface, resolve the bsdf-sampled
		// continuation straight against the environment -- no AS trace,
		// so e.g. a mirror sphere reflects the env map in place of a
		// neighboring object instead of that ray either cross-lighting
		// the scene or (if just discarded) leaving a black hole where a
		// real trace would have hit geometry
		if (unf.envMapOnlyMode != 0u && i > 0) {
			const f32v3 radiance = (
				sampleEnvironment(
					/*dir=*/ bsdfWo,
					/*intensity=*/ unf.envIntensity,
					/*envMode=*/ unf.envMode,
					/*envHandles=*/ envHandles,
					/*envRotation=*/ unf.envRotation
				)
			);
			if (radiance != f32v3(0.0f)) {
				f32 weightB = 1.0f;
				if (envNeeActive) {
					const f32 envPdf = (
						environmentMapPdf(envHandles, bsdfWo, unf.envRotation)
					);
					weightB = misBalanceWeight(bsdfPdf, envPdf);
				}
				irradianceAccumulator += radiance * irradianceThroughput * weightB;
				hit = 1;
			} else if (any(greaterThan(irradianceAccumulator, f32v3(0.0f)))) {
				hit = 1;
			}
			break;
		}

		bool mediumScatter;
		f32 phasePdf;
		const int walkEnum = (
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
				state,
				unf,
				envHandles,
				envNeeActive
			)
		);
		// TODO(spp validity): a medium scatter alone should count as a
		// valid sample; skWalkCap paths currently only report hit when
		// something was already accumulated
		if (walkEnum == skWalkCap) {
			if (any(greaterThan(irradianceAccumulator, f32v3(0.0f)))) {
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
			if (radiance != f32v3(0.0f)) {
				// mis-weight heuristics for bsdf-sampled contribution
				// against nee's estimate of the same env direction
				f32 weightB = 1.0f;
				if (envNeeActive) {
					const f32 envPdf = (
						environmentMapPdf(envHandles, bsdfWo, unf.envRotation)
					);
					if (mediumScatter) {
						// heuristic for phase-sampled escape
						weightB = misBalanceWeight(phasePdf, envPdf);
					} else if (i > 0) {
						// heuristic for bsdf-sampled escape
						weightB = misBalanceWeight(bsdfPdf, envPdf);
					}
				}
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
			} else if (any(greaterThan(irradianceAccumulator, f32v3(0.0f)))) {
				// a black environment still ends the path; whatever emission
				// was accumulated on the way here is the path's contribution
				hit = 1;
			}
			break;
		}

		// -- surface vertex
		const int returnEnum = (
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
				envNeeActive
			)
		);

		if (returnEnum == skHitIndirect) {
			hit = 1;
		}

		if (returnEnum == skHitTerminate) {
			// terminate still permits using whatever was already
			// accumulated (e.g. emission seen before an invalid bsdf
			// sample cut the path short); only a bare miss reports no hit
			if (any(greaterThan(irradianceAccumulator, f32v3(0.0f)))) {
				hit = 1;
			}
			break;
		}
	}

	// debug for NaN hunts
	if (any(isnan(irradianceAccumulator)) && hit > 0) {
		return f32v4(9999.0f, 0.0f, 0.0f, 1.0f);
	}

	return f32v4(irradianceAccumulator, float(hit));
}

#endif // UTIL_RAYTRACE_GLSL
