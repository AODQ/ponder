// util to trace rays in the scene using ray queries

#ifndef UTIL_RAYTRACE_GLSL
#define UTIL_RAYTRACE_GLSL

#include "util-mesh.glsl"

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
				rayQueryGetIntersectionBarycentricsEXT(rq, false)
			)
		);
		f32v3 modelNormal;
		f32v4 tangentNormal;
		f32v3 modelCoatNormal;
		f32v4 modelClearcoatNormal;
		const OpenPbrMaterial mat = (
			openPbrLoadMaterialLod(
				/*mat=*/ rqData.material,
				/*uvTransforms=*/rqData.uvTransforms,
				/*uv=*/rqData.uv,
				/*lod=*/ 0,
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

// a direct contribution was found. The path is terminated and the
// accumulated irradiance can be used. No further bouncing.
#define skHitDirect 1

// an indirect contribution was found. The path can continue bouncing, and
// on path termination the accumulated irradiance can be used.
#define skHitIndirect 2

// the path should be terminated. Any contribution can be used however
// if there is none accumulated, then no overall contribution.
#define skHitTerminate 3

struct Ray {
	f32v3 origin;
	f32v3 dir;
};

int utilIrradiance(
	const MaterialTableHandles tables,
	const OpenPbrMaterial material,
	inout f32v3 itOri,
	inout ShadingFrame itFrame,
	inout ShadingFrame itCoatFrame,
	inout f32v3 itNormalGeometrical,
	inout f32v3 itWi,
	inout f32v3 irradianceThroughput,
	inout f32v3 irradianceAccumulator,
	out f32v3 bsdfWo,
	out Ray rayWo,
	inout u64 state,
	out RayQueryResult bsdfRq,
	out bool sampledTransmission,
	const int iteration,
	const bool isInsideMedium,
	// -1 = no hero wavelength locked yet (path hasn't sampled a dispersive
	// transmission event); once locked (0/1/2 = R/G/B), stays locked for
	// the rest of the path -- see the dispersion resolution block below
	inout int pathSpectralChannel
) {
	int returnEnum = skHitNone;
	f32 bsdfPdf;

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

	// -- indirect irradiance propagation (bsdf path)
	{
		bsdfRq = (
			utilTraceRay(
				itOri + itNormalGeometrical * (sampledTransmission ? -0.0005f : 0.0005f),
				bsdfWo,
				/*maxDist=*/999999.0f,
				/*isShadowRay=*/false,
				/*testAlphaCutoff=*/true
			)
		);

		// -- environment contribution; a miss always terminates the path
		if (bsdfRq.dist <= 0.0f) {
			const GpuDebugPC unf = GpuDebugPCBuffer(pc.global.debug).data;
			const f32v3 radiance = (
				sampleEnvironment(
					/*dir=*/ bsdfWo,
					/*intensity=*/ unf.envIntensity,
					/*envMode=*/ unf.envMode
				)
			);
			if (radiance != f32v3(0.0f)) {
				irradianceAccumulator += radiance * irradianceThroughput;
				return skHitDirect;
			}
			// a black environment still ends the path; whatever emission was
			// accumulated on the way here is the path's contribution
			return skHitTerminate;
		}
	}

	return returnEnum;
}

int utilIrradiancePropagate(
	const MaterialTableHandles tables,
	inout OpenPbrMaterial material,
	inout f32v3 itOri,
	inout ShadingFrame itFrame,
	inout ShadingFrame itCoatFrame,
	inout f32v3 itNormalGeometrical,
	inout f32v3 itWi,
	inout f32v3 irradianceThroughput,
	inout f32v3 irradianceAccumulator,
	inout u64 state,
	const int iteration,
	inout bool isInsideMedium,
	inout int pathSpectralChannel
) {
	// -- compute irradiance on surface
	f32v3 bsdfWo;
	Ray rayWo;
	RayQueryResult bsdfRq;
	bool sampledTransmission;
	int returnEnum = (
		utilIrradiance(
			tables,
			material,
			itOri,
			itFrame,
			itCoatFrame,
			itNormalGeometrical,
			itWi,
			irradianceThroughput,
			irradianceAccumulator,
			bsdfWo,
			rayWo,
			state,
			bsdfRq,
			sampledTransmission,
			iteration,
			isInsideMedium,
			pathSpectralChannel
		)
	);
	if (sampledTransmission) {
		isInsideMedium = !isInsideMedium;
	}

	// -- Beer-Lambert attenuation through transmissive medium; a thin-walled
	// shell has no real interior volume to traverse, so this only applies
	// to solid dielectrics (khr_materials_volume with real thickness)
	if (
		sampledTransmission
		&& isInsideMedium
		&& material.geometryThinWalled <= 0.0f
		&& material.transmissionDepth > 0.0f
		&& bsdfRq.dist > 0.0f
		&& returnEnum != skHitTerminate
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
		irradianceThroughput *= exp(-mu * bsdfRq.dist);
		// transmissionColor components near 1.0 make mu near 0, and dist can
		// be large for a long chord through the medium -- exp(-mu*dist) is
		// always finite on its own, but the same Inf-then-times-zero
		// overflow risk as the bsdf divide above applies once this
		// multiplies into an already-large throughput
		NAN_CHECK3(
			irradianceThroughput,
			"NaN utilIrradiancePropagate.irradianceThroughput after beer-lambert px(%d,%d)=%v3f"
		)
	}

	// -- update the material and ray for the next bounce
	if (returnEnum == skHitIndirect || returnEnum == skHitNone) {
		UtilMeshAttributeDataFromIndices rqData = (
			utilMeshAttributeDataFromIndices(
				bsdfRq.modelDrawIndex,
				bsdfRq.primitiveIndex,
				bsdfRq.barycentric
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
			openPbrLoadMaterialLod(
				/*mat=*/ rqData.material,
				/*uvTransforms=*/rqData.uvTransforms,
				/*uv=*/rqData.uv,
				/*lod=*/ 0,
				modelNormal, tangentNormal, modelCoatNormal, modelClearcoatNormal
			)
		);
		itOri = itOri + bsdfWo * bsdfRq.dist;
		// shading frame from the normal-mapped normal + mesh/uv tangent
		itFrame = (
			shadingFrameFromTangent(
				normalize(tbn * modelNormal), rqData.tangent.xyz
			)
		);
		// coat's own shading normal (khr_materials_clearcoat), same mesh
		// tangent basis as the base frame above
		itCoatFrame = (
			shadingFrameFromTangent(
				normalize(tbn * modelCoatNormal), rqData.tangent.xyz
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

	return returnEnum;
}

f32v4 utilIrradiancePropagationEntry(
	const MaterialTableHandles tables,
	OpenPbrMaterial originalMaterial,
	f32v3 origin,
	ShadingFrame frame,
	ShadingFrame coatFrame,
	f32v3 normalGeometrical,
	f32v3 wi,
	const uint propagationDepth,
	inout u64 state
) {
	f32v3 irradianceThroughput = f32v3(1.0);
	f32v3 irradianceAccumulator = f32v3(0.0);
	bool isInsideMedium = false;
	// -1: no hero wavelength locked yet; see the dispersion resolution
	// block in utilIrradiance
	int pathSpectralChannel = -1;

	int hit = 0;
	for (int i = 0; i < propagationDepth; ++i) {
		int returnEnum = (
			utilIrradiancePropagate(
				tables,
				originalMaterial,
				origin,
				frame,
				coatFrame,
				normalGeometrical,
				wi,
				irradianceThroughput,
				irradianceAccumulator,
				state,
				i,
				isInsideMedium,
				pathSpectralChannel
			)
		);

		if (returnEnum == skHitNone) {
			continue;
		}

		if (returnEnum == skHitIndirect) {
			hit = 1;
			continue;
		}

		if (returnEnum == skHitDirect) {
			hit = 1;
			break;
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
