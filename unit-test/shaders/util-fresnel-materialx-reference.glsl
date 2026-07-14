#ifndef UTIL_FRESNEL_MATERIALX_REFERENCE_GLSL
#define UTIL_FRESNEL_MATERIALX_REFERENCE_GLSL

#ifndef f32
#define f32 float
#define f32v2 vec2
#define f32v3 vec3
#define u32 uint
#endif

// independent cross-check only, not used by the real ponder implementation.
// ported from AcademySoftwareFoundation/MaterialX
// libraries/pbrlib/genglsl/lib/mx_microfacet_specular.glsl

// seb lagarde's "g" formulation -- different algebra than
// utilFresnelDielectric's snell's-law form, same physics
f32 mxFresnelDielectric(f32 cosTheta, f32 ior) {
	const f32 c = cosTheta;
	const f32 g2 = ior*ior + c*c - 1.0f;
	if (g2 < 0.0f) {
		return 1.0f; // total internal reflection
	}
	const f32 g = sqrt(g2);
	const f32 a = (g - c) / (g + c);
	const f32 b = ((g + c) * c - 1.0f) / ((g - c) * c + 1.0f);
	return 0.5f * (a*a) * (1.0f + b*b);
}

// hoffman's generalization of f82-tint; F90=white, exponent=5, f82=
// specularWeight*specularColor reduces to exactly openpbr's formula
f32v3 mxFresnelHoffmanSchlick(
	f32 cosTheta, f32v3 f0, f32v3 f82, f32v3 f90, f32 exponent
) {
	const f32 kCosThetaMax = 1.0f / 7.0f;
	const f32 kCosThetaFactor = (
		1.0f / (kCosThetaMax * pow(1.0f - kCosThetaMax, 6.0f))
	);
	const f32 x = clamp(cosTheta, 0.0f, 1.0f);
	const f32v3 a = (
		mix(f0, f90, pow(1.0f - kCosThetaMax, exponent))
		* (f32v3(1.0f) - f82) * kCosThetaFactor
	);
	return mix(f0, f90, pow(1.0f - x, exponent)) - a * x * pow(1.0f - x, 6.0f);
}

#endif
