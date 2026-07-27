#ifndef UTIL_DIELECTRIC_SPECULAR_GLSL
#define UTIL_DIELECTRIC_SPECULAR_GLSL

#ifndef f32
#define i32v2 ivec2
#define i32 int
#define f32 float
#define f32v2 vec2
#define f32v3 vec3
#define f32v4 vec4
#define f32m44 mat4
#define u32 uint
#define u32v2 uvec2
#define u32v3 uvec3
#define u32v4 uvec4
#define u64 uint64_t
#endif

// microfacet dielectric interface reflection, shared by the opaque base and
// the translucent base.
//
// ported from cull's openPbrDielectricSpecularEvaluateF, with two
// deliberate deviations from a strict 1:1 port:
//   1. takes the material fields it needs directly instead of an
//      OpenPbrMaterial struct, which doesn't exist in ponder yet (same
//      adaptation as utilFresnelMetallic earlier in this port).
//   2. thin-film interference is scoped OUT for now (a separate,
//      substantial feature -- its own file in cull, its own conditional
//      branch here); mfFresnel is always the plain dielectric fresnel
//      term. cull's thinFilmWeight>0 branch is omitted, not stubbed.
/*
	f_{dielectric} =
		mf_f \cdot mf_d \cdot mf_v
	\\
	where \eta = relative ior with the specular weight folded in, and the
	1/(4 (\omega_i \cdot n)(\omega_o \cdot n)) term is already folded into
	mf_v (see utilMicrofacetSmithGgxVisibilityAniso)
*/
f32v3 openPbrDielectricSpecularEvaluateF(
	const f32 specularRoughness,
	const f32 specularRoughnessAnisotropy,
	const f32 specularRoughnessAnisotropyRotation,
	const f32 coatRoughness,
	const f32 coatWeight,
	const ShadingFrame frame,
	const f32v3 wi,
	const f32v3 wo,
	const f32 etaRel
) {
	// per-axis slopes from the openpbr anisotropy mapping; the base
	// roughness is widened by a rough coat (openpbr spec, coat roughening)
	const f32 roughenedRoughness = (
		openPbrCoatRoughenedRoughness(specularRoughness, coatRoughness, coatWeight)
	);
	const f32v2 alpha = (
		openPbrRoughnessAlpha(roughenedRoughness, specularRoughnessAnisotropy)
	);
	// gltf interop anisotropy rotation (not part of openpbr)
	const ShadingFrame specFrame = (
		shadingFrameRotate(frame, specularRoughnessAnisotropyRotation)
	);
	// local-frame directions for the anisotropic D and V
	const f32v3 wiLocal = (
		f32v3(
			dot(wi, specFrame.tanX),
			dot(wi, specFrame.tanY),
			dot(wi, specFrame.nor)
		)
	);
	const f32v3 woLocal = (
		f32v3(
			dot(wo, specFrame.tanX),
			dot(wo, specFrame.tanY),
			dot(wo, specFrame.nor)
		)
	);
	// wo landing exactly at -wi (a retroreflection) makes wiLocal + woLocal
	// the zero vector, whose normalize() is a NaN that then poisons every
	// term below -- same guard as this function's production counterpart,
	// openPbrDielectricSpecularEvaluateF in util-material-openpbr-
	// microfacet.glsl (see that guard's comment for the eta==1 Inf
	// mechanism this function's Fresnel call shares); mfVisibility below
	// does correctly zero for a non-same-side wi/wo, but NaN * 0 is still
	// NaN, not 0
	const f32v3 hLocalUnnorm = wiLocal + woLocal;
	if (dot(hLocalUnnorm, hLocalUnnorm) == 0.0f) {
		return f32v3(0.0f);
	}
	const f32v3 hLocal = normalize(hLocalUnnorm);
	// dot(h, wi) is frame-invariant
	const f32 dotHWi = dot(hLocal, wiLocal);
	// a non-positive dotHWi is a back-facing (physically invalid) microfacet
	// for this wi -- zero reflectance, matching the reference guard
	if (dotHWi <= 0.0f) {
		return f32v3(0.0f);
	}
	const f32v3 mfFresnel = f32v3(utilFresnelDielectric(dotHWi, etaRel));
	const f32 mfDistribution = utilMicrofacetGgxDistributionAniso(hLocal, alpha);
	const f32 mfVisibility = (
		utilMicrofacetSmithGgxVisibilityAniso(wiLocal, woLocal, alpha)
	);
	// turquin 2019 multiplicative energy compensation; f0 from the relative
	// ior, which is symmetric in \eta vs 1/\eta. the factor uses the view
	// cosine only. the fit is isotropic, so it is indexed with the
	// area-preserving effective roughness \sqrt{\alpha_x \alpha_y}
	const f32 iorRatio = (etaRel - 1.0f) / (etaRel + 1.0f);
	const f32 f0 = iorRatio * iorRatio;
	const f32 energyCompensation = (
		utilMicrofacetDielectricEnergyCompensate(
			wiLocal.z, sqrt(alpha.x * alpha.y), f0
		)
	);
	return mfFresnel * mfDistribution * mfVisibility * energyCompensation;
}

#endif // UTIL_DIELECTRIC_SPECULAR_GLSL
