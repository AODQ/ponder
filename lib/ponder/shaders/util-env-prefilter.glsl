#ifndef UTIL_ENV_PREFILTER_GLSL
#define UTIL_ENV_PREFILTER_GLSL

#ifndef TAU
#define TAU 6.28318530717958647692
#endif

// -----------------------------------------------------------------------------
// -- envPrefilterRadicalInverseVdc
// -----------------------------------------------------------------------------

float envPrefilterRadicalInverseVdc(uint bits) {
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return float(bits) * 2.3283064365386963e-10f;
}

vec2 envPrefilterHammersley(const uint i, const uint n) {
	return vec2(float(i) / float(n), envPrefilterRadicalInverseVdc(i));
}

// karis 2013, importance-samples the ggx half-vector directly
// alpha is linear roughness
vec3 envPrefilterImportanceSampleH(
	const vec2 xi, const float alpha, const vec3 nor
) {
	const float phi = TAU * xi.x;
	const float a2 = alpha * alpha;
	const float cosTheta = sqrt((1.0f - xi.y) / (1.0f + (a2 - 1.0f) * xi.y));
	const float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
	const vec3 hLocal = (
		vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta)
	);
	vec3 tangent, bitangent;
	utilCalculateXy(nor, tangent, bitangent);
	return normalize(tangent * hLocal.x + bitangent * hLocal.y + nor * hLocal.z);
}

// -----------------------------------------------------------------------------
// -- envPrefilterComputeLod
// -----------------------------------------------------------------------------

// gpu gems 3 20.4 eq 13 (krivanek & colbert 2008): picks a pre-blurred
// source mip proportional to the sample's own solid angle, so a sample
// aimed near a tiny bright feature (e.g. a sun disc) reads an
// already-averaged value instead of either its full peak or nothing
float envPrefilterComputeLod(
	const vec3 dir, const float pdf, const float maxMipLevel, const uint sampleCount
) {
	const float mipOffset = 1.5f;
	const float effectiveMaxMip = maxMipLevel - mipOffset;
	const float distortion = sqrt(max(0.0f, 1.0f - dir.y * dir.y));
	const float solidAngleTerm = float(sampleCount) * pdf * distortion;
	// glsl's log2 is undefined for x <= 0 (not guaranteed -inf); this hits
	// exactly 0 at alpha=0 (a perfect mirror) or dir.y=+-1 (map poles),
	// which should read the finest mip anyway -- skip log2 entirely rather
	// than rely on it clamping back through max()
	if (solidAngleTerm <= 0.0f) {
		return 0.0f;
	}
	return max(effectiveMaxMip - 0.5f * log2(solidAngleTerm), 0.0f);
}

#endif // UTIL_ENV_PREFILTER_GLSL
