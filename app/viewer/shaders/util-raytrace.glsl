// util to trace rays in the scene using ray queries

#ifndef UTIL_RAYTRACE_GLSL
#define UTIL_RAYTRACE_GLSL

#include "util-mesh.glsl"

#include "util-vdb.glsl"
#include "util-material-openpbr.glsl"
#include "util-material-openpbr-load.glsl"

// interpolated shading normals can degenerate in real assets: a mesh with
// no NORMAL attribute loads as all-zero normals (fox, lpshead), and a
// triangle whose corner normals are antiparallel interpolates through
// length zero (bistro). normalize() of either is NaN and poisons every
// lobe sampled from the resulting frame, so fall back to the geometric
// normal -- which for a missing NORMAL attribute is exactly the flat
// normal the gltf spec mandates. the test is written so a NaN length
// fails into the fallback too
f32v3 utilShadingNormalOrGeometric(
	const f32v3 shadingNormalRaw,
	const f32v3 normalGeometrical
) {
	const float len2 = dot(shadingNormalRaw, shadingNormalRaw);
	if (len2 > 1e-12f) {
		return shadingNormalRaw * inversesqrt(len2);
	}
	return normalGeometrical;
}

// -----------------------------------------------------------------------------
// -- ray-query utility
// -----------------------------------------------------------------------------

struct RayQueryResult {
	float dist;
	uint modelDrawIndex;
	uint primitiveIndex;
	f32v2 barycentric;
};

RayQueryResult utilTraceRay(
	const vec3 origin,
	const vec3 dir,
	const float maxDist,
	const bool isShadowRay,
	const bool testAlphaCutoff
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
				/*mat=*/ rqData.material,
				/*uvTransforms=*/rqData.uvTransforms,
				/*uv=*/rqData.uv,
				/*uvDx=*/rqData.uvDx,
				/*uvDy=*/rqData.uvDy,
				modelNormal, tangentNormal, modelCoatNormal, modelClearcoatNormal
			)
		);

		if (mat.geometryOpacity > mat.alphaCutoff) {
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
	// n(\lambda_c); falls back to the flat mat.specularIor when this
	// material has no dispersion at all
	const f32 dispersedIor = (
		material.transmissionDispersionScale > 0.0f
			? openPbrDispersionIorRgb(material)[provisionalChannel]
			: material.specularIor
	);

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
						/*testAlphaCutoff=*/true
					)
				);
				if (visibilityRq.dist <= 0.0f) {
					const f32 transmittance = (
						utilVdbFogTransmittance(
							vdb, shadowOrigin, envDir,
							unf.vdbSigmaScale, unf.fogThickness,
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
	u32 pickedLobeUnused;
	bsdfWo = (
		openPbrSampleWo(
			/*tables=*/ tables,
			/*frame=*/ itFrame,
			/*coatFrame=*/ itCoatFrame,
			/*wi=*/ itWi,
			/*mat=*/ material,
			/*pdf=*/ bsdfPdf,
			/*sampledTransmission=*/ sampledTransmission,
			/*pickedLobe=*/ pickedLobeUnused,
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
	if (!sampledTransmission && dotNorWo <= 0.0f) {
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
	itOri += itNormalGeometrical * (sampledTransmission ? -0.0005f : 0.0005f);

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
	if (sampledTransmission) {
		isInsideMedium = !isInsideMedium;
	}
	// beer-lambert attenuation for the entered dielectric interior moved to
	// utilIrradianceWalk, which owns the next segment and its length

	return returnEnum;
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
			/*testAlphaCutoff=*/true
		)
	);

	// -- Beer-Lambert attenuation through transmissive medium (moved here
	// from utilIrradiancePropagate: this walk owns the segment length); a
	// thin-walled shell has no real interior volume to traverse, so this
	// only applies to solid dielectrics (khr_materials_volume with real
	// thickness). vdb scatter inside a dielectric interior is not handled:
	// attenuation uses the full straight segment
	if (
		sampledTransmission
		&& isInsideMedium
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
			utilVdbPhaseFogDefaultParams(unf.vdbDropletDiameter)
		);
		for (i32 s = 0; s < 64; ++s) {
			const f32 distMax = rq.dist > 0.0f ? rq.dist : 999999.0f;
			const f32 tScatterVdb = (
				vdb.isValid
				? utilVdbDeltaTrack(
					vdb, itOri, bsdfWo, distMax, unf.vdbSigmaScale, state
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
			irradianceThroughput *= fogWins ? unf.fogAlbedo : unf.vdbAlbedo;

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
							/*testAlphaCutoff=*/true
						)
					);
					if (visibilityRq.dist <= 0.0f) {
						const f32 transmittance = (
							utilVdbFogTransmittance(
								vdb, itOri, envDir,
								unf.vdbSigmaScale, unf.fogThickness,
								unf.fogDistanceMax, state
							)
						);
						if (transmittance > 0.0f) {
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
					/*testAlphaCutoff=*/true
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
	{
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
		f32v3 modelNormal;
		f32v4 tangentNormal;
		f32v3 modelCoatNormal;
		f32v4 modelClearcoatNormal;
		material = (
			openPbrLoadMaterialDeriv(
				/*mat=*/ rqData.material,
				/*uvTransforms=*/rqData.uvTransforms,
				/*uv=*/rqData.uv,
				/*uvDx=*/rqData.uvDx,
				/*uvDy=*/rqData.uvDy,
				modelNormal, tangentNormal, modelCoatNormal, modelClearcoatNormal
			)
		);
		itOri = itOri + bsdfWo * rq.dist;
		// shading frame from the normal-mapped normal + mesh/uv tangent
		itFrame = (
			shadingFrameFromTangent(
				utilShadingNormalOrGeometric(
					tbn * modelNormal, rqData.normalGeometrical
				),
				rqData.tangent.xyz
			)
		);
		// coat's own shading normal (khr_materials_clearcoat), same mesh
		// tangent basis as the base frame above
		itCoatFrame = (
			shadingFrameFromTangent(
				utilShadingNormalOrGeometric(
					tbn * modelCoatNormal, rqData.normalGeometrical
				),
				rqData.tangent.xyz
			)
		);
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
