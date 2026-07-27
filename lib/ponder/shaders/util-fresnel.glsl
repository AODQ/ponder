#ifndef UTIL_FRESNEL_GLSL
#define UTIL_FRESNEL_GLSL

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

// dielectric fresnel; eta = specularIor for reflection, 1/specularIor for
// refraction. cosTheta must be >= 0. ported from cull's
// utilMicrofacetFresnelDielectric.
f32 utilFresnelDielectric(const f32 cosTheta, const f32 eta) {
	const f32 cosTheta2 = cosTheta * cosTheta;
	const f32 sinTheta2 = 1.0f - cosTheta2;
	const f32 eta2 = eta * eta;
	const f32 sinThetaT2 = sinTheta2 / eta2;
	if (sinThetaT2 > 1.0f) {
		return 1.0f; // total internal reflection
	}
	const f32 cosThetaT = sqrt(1.0f - sinThetaT2);
	// at cosTheta=0, eta=1 exactly (grazing + matched ior) both
	// denominators go to exactly 0/0; epsilon avoids the nan, the true
	// value there is a genuine discontinuity (r=0 along eta=1, r=1 along
	// cosTheta=0 from any other eta) with no single continuous limit
	const f32 rParallel = (
		(eta * cosTheta - cosThetaT) / max(eta * cosTheta + cosThetaT, 1e-7f)
	);
	const f32 rPerpendicular = (
		(cosTheta - eta * cosThetaT) / max(cosTheta + eta * cosThetaT, 1e-7f)
	);
	return 0.5f * (rParallel*rParallel + rPerpendicular*rPerpendicular);
}

// f82-tint conductor fresnel (kutz, hasan & edmondson 2021, "novel aspects
// of the adobe standard material"), openpbr spec "metal" section. dotNorWi
// must be >= 0. ported from cull's openPbrFresnelMetallicEvaluateF
// (fresnel term only -- D/V/energy-compensation are separate, later).
f32v3 utilFresnelMetallic(
	const f32v3 f0,
	const f32 specularWeight,
	const f32v3 specularColor,
	const f32 dotNorWi
) {
	// F_s(1/7); (1 - 1/7)^5 = 0.462664
	const f32v3 fresnelConstS = f0 + (f32v3(1.0f) - f0) * 0.462664f;
	const f32v3 fresnelConst = (
		fresnelConstS * (f32v3(1.0f) - specularWeight * specularColor)
	);
	// (1/7)(1 - 1/7)^6 = 0.056653
	const f32 fresnelDenom = 0.056653f;

	const f32v3 fs = f0 + (f32v3(1.0f) - f0) * pow(1.0f - dotNorWi, 5.0f);
	// clamped at zero: the raw f82 correction dips negative for dark f0
	// combined with low specularWeight*specularColor (the model assumes
	// metal-like f0). deliberate deviation from cull (which returns the
	// raw value), user decision 2026-07-14; keep in sync with the inline
	// copy in util-material-openpbr-metallic.glsl
	return (
		max(
			fs
			- (
				(dotNorWi * pow(1.0f - dotNorWi, 6.0f) / fresnelDenom)
				* fresnelConst
			),
			f32v3(0.0f)
		)
	);
}

#endif // UTIL_FRESNEL_GLSL
