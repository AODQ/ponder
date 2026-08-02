module "util-env-prefilter";
__exported import "util-types";
__exported import "util-shading-frame";

#ifndef TAU
#endif

public func envPrefilterRadicalInverseVdc(
	bits: u32,
) -> f32 {
	// glsl parameters are mutable copies; slang's are not
	var bits = bits;
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return f32(bits) * 2.3283064365386963e-10f;
}

public func envPrefilterHammersley(
	i: u32,
	n: u32,
) -> f32v2 {
	return f32v2(f32(i) / f32(n), envPrefilterRadicalInverseVdc(i));
}

// karis 2013, importance-samples the ggx half-vector directly
// alpha is linear roughness
public func envPrefilterImportanceSampleH(
	xi: f32v2,
	alpha: f32,
	nor: f32v3,
) -> f32v3 {
	const f32 phi = TAU * xi.x;
	const f32 a2 = alpha * alpha;
	const f32 cosTheta = sqrt((1.0f - xi.y) / (1.0f + (a2 - 1.0f) * xi.y));
	const f32 sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
	const f32v3 hLocal = (
		f32v3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta)
	);
	f32v3 tangent, bitangent;
	utilCalculateXy(nor, tangent, bitangent);
	return normalize(tangent * hLocal.x + bitangent * hLocal.y + nor * hLocal.z);
}

// gpu gems 3 20.4 eq 13 (krivanek & colbert 2008): picks a pre-blurred
// source mip proportional to the sample's own solid angle, so a sample
// aimed near a tiny bright feature (e.g. a sun disc) reads an
// already-averaged value instead of either its full peak or nothing
public func envPrefilterComputeLod(
	dir: f32v3,
	pdf: f32,
	maxMipLevel: f32,
	sampleCount: u32,
) -> f32 {
	const f32 mipOffset = 1.5f;
	const f32 effectiveMaxMip = maxMipLevel - mipOffset;
	const f32 distortion = sqrt(max(0.0f, 1.0f - dir.y * dir.y));
	const f32 solidAngleTerm = f32(sampleCount) * pdf * distortion;
	// glsl's log2 is undefined for x <= 0 (not guaranteed -inf); this hits
	// exactly 0 at alpha=0 (a perfect mirror) or dir.y=+-1 (map poles),
	// which should read the finest mip anyway -- skip log2 entirely rather
	// than rely on it clamping back through max()
	if (solidAngleTerm <= 0.0f) {
		return 0.0f;
	}
	return max(effectiveMaxMip - 0.5f * log2(solidAngleTerm), 0.0f);
}

