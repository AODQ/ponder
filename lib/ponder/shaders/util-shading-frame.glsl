#ifndef UTIL_SHADING_FRAME_GLSL
#define UTIL_SHADING_FRAME_GLSL

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

#include "util-random.glsl"

void utilCalculateXy(
	const f32v3 nor,
	out f32v3 binormal,
	out f32v3 bitangent
) {
	// frisvad duff 2017 orthornormal basis revisted
	const f32 sgn = nor.z >= 0.0f ? 1.0f : -1.0f;
	const f32 a = -1.0f / (sgn + nor.z);
	const f32 b = nor.x * nor.y * a;
	binormal = f32v3(1.0f + sgn*nor.x*nor.x*a, sgn*b, -sgn*nor.x);
	bitangent = f32v3(b, sgn + nor.y*nor.y*a, -nor.y);
}

mat3 utilCalculateTbnBasis(
	const vec3 nor,
	const vec4 tangent
) {
	// gram-schmidt re-orthogonalization
	const vec3 n = normalize(nor);
	const vec3 tRaw = tangent.xyz - n * dot(n, tangent.xyz);
	const float tLen = length(tRaw);
	if (tLen < 1e-6) {
		// degerate tangent, fall-back to Frisvad basis
		vec3 binormal, bitangent;
		utilCalculateXy(n, binormal, bitangent);
		return mat3(binormal, bitangent, n);
	}
	const vec3 t = tRaw / tLen;
	const vec3 bitangent = (cross(n, t) * tangent.w);
	return mat3(t, bitangent, n);
}

// shading frame for the anisotropic lobes; tanY = cross(nor, tanX).
// bitangent handedness does not matter distribution-wise: the aniso ggx D
// and smith lambda are even in both tangent axes, and the cap sampler's
// azimuth is distribution-uniform
struct ShadingFrame {
	f32v3 tanX;
	f32v3 tanY;
	f32v3 nor;
};

// frisvad fallback frame; rotationally arbitrary around nor, so only valid
// for isotropic lobes or when no mesh tangent exists
ShadingFrame shadingFrameFromNormal(const f32v3 nor) {
	f32v3 binormal, bitangent;
	utilCalculateXy(nor, binormal, bitangent);
	ShadingFrame frame;
	frame.tanX = bitangent;
	frame.tanY = binormal;
	frame.nor = nor;
	return frame;
}

// shading frame from the mesh/uv tangent, re-orthogonalized against the
// (possibly normal-mapped) shading normal
ShadingFrame shadingFrameFromTangent(const f32v3 nor, const f32v3 tangent) {
	const f32v3 tRaw = tangent - nor * dot(nor, tangent);
	const f32 tLen2 = dot(tRaw, tRaw);
	if (tLen2 < 1e-10f) {
		// degenerate tangent: fall back to the frisvad frame
		return shadingFrameFromNormal(nor);
	}
	ShadingFrame frame;
	frame.tanX = tRaw * inversesqrt(tLen2);
	frame.tanY = cross(nor, frame.tanX);
	frame.nor = nor;
	return frame;
}

// falls back to the geometric normal on a degenerate (zero/NaN-length)
// shading normal, e.g. a missing NORMAL attribute or antiparallel corners
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

// rotates the tangent axes around nor; used for the gltf interop
// anisotropy rotation (khr_materials_anisotropy), which openpbr itself
// does not have. counter-clockwise from the tangent toward the bitangent
/*
	t' = \cos(\phi) t + \sin(\phi) b\\
	b' = \cos(\phi) b - \sin(\phi) t
*/
ShadingFrame shadingFrameRotate(const ShadingFrame frame, const f32 rotation) {
	if (rotation == 0.0f) {
		return frame;
	}
	const f32 c = cos(rotation);
	const f32 s = sin(rotation);
	ShadingFrame rotated;
	rotated.tanX = c * frame.tanX + s * frame.tanY;
	rotated.tanY = c * frame.tanY - s * frame.tanX;
	rotated.nor = frame.nor;
	return rotated;
}

vec3 utilReorientHemisphere(vec3 wo, vec3 nor) {
	vec3 binormal, bitangent;
	const vec3 n = normalize(nor);
	utilCalculateXy(n, binormal, bitangent);
	return bitangent*wo.x + binormal*wo.y + wo.z*n;
}

vec3 utilToCartesian(const float cosTheta, const float phi) {
	const float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta*cosTheta));
	return vec3(cos(phi)*sinTheta, sin(phi)*sinTheta, cosTheta);
}

vec3 utilCosineHemisphereSampleWo(
	vec3 nor, const vec2 xi
) {
	const vec3 wo = (
		utilReorientHemisphere(
			normalize(utilToCartesian(sqrt(xi.x), skTau*xi.y)),
			nor
		)
	);
	return wo;
}

vec3 utilDirectionAboutAxis(
	const vec3 axis, const float cosTheta, const float phi
) {
	const ShadingFrame frame = shadingFrameFromNormal(axis);
	const float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta*cosTheta));
	return (
		frame.tanX * (sinTheta * cos(phi))
		+ frame.tanY * (sinTheta * sin(phi))
		+ frame.nor * cosTheta
	);
}

#endif // UTIL_SHADING_FRAME_GLSL
