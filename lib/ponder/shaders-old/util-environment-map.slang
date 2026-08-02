module "util-environment-map";
__exported import "util-types";


#ifndef PI
#endif

// HDR environment map precomputed importance-sampling tables.
// radiance is r32g32b32a32 float WxH
// pdf is r32 float WxH
// aliasProb/aliasIndex are r32 float WxH, Vose's alias method -- O(1)
// sampling of the same distribution pdf describes, see
// environmentMapImportanceSample
public struct EnvironmentMapHandles {
	public u32 radiance;
	public u32 pdf;
	public u32 aliasProb;
	public u32 aliasIndex;
};


// world-space direction to equirect
public func environmentMapDirToUv(
	dir: f32v3,
	rotation: f32,
) -> f32v2 {
	const f32 theta = acos(clamp(dir.y, -1.0f, 1.0f));
	const f32 phi = atan2(dir.z, dir.x);
	const f32 u = frac(0.5f + phi / (2.0f * PI) + rotation / (2.0f * PI));
	const f32 v = theta / PI;
	return f32v2(u, v);
}

// equirect uv to world-space direction; inverse of environmentMapDirToUv
public func environmentMapUvToDir(
	uv: f32v2,
	rotation: f32,
) -> f32v3 {
	const f32 theta = uv.y * PI;
	const f32 phi = (uv.x - 0.5f - rotation / (2.0f * PI)) * 2.0f * PI;
	const f32 sinTheta = sin(theta);
	const f32 cosTheta = cos(theta);
	return f32v3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
}

// solid-angle pdf of environment map's importance distribution at 'dir'
public func environmentMapPdf(
	handles: EnvironmentMapHandles,
	dir: f32v3,
	rotation: f32,
) -> f32 {
	const f32v2 uv = environmentMapDirToUv(dir, rotation);
	const f32 sinTheta = sin(uv.y * PI);
	if (sinTheta <= 1e-6f) {
		return 0.0f;
	}
	// uv to discrete texel coordinate
	const i32v2 pdfSize = textureSize(vkofTextures[handles.pdf], 0);
	const i32v2 texel = i32v2(
		clamp(i32(uv.x * f32(pdfSize.x)), 0, pdfSize.x - 1),
		clamp(i32(uv.y * f32(pdfSize.y)), 0, pdfSize.y - 1)
	);
	// convert to continuous uv-space density then multiply by the
	// equirect solid-angle jacobian
	const f32 pdfTexel = texelFetch(vkofTextures[handles.pdf], texel, 0).r;
	const f32 pdfUv = pdfTexel * f32(pdfSize.x) * f32(pdfSize.y);
	return pdfUv / (2.0f * PI * PI * sinTheta);
}

// generate an importance-sampled direction from the environment map's
// precomputed distribution via Vose's alias method: one O(1) lookup (two
// independent texel reads, no dependent search chain) instead of a
// marginal+conditional cdf binary search. xi.x picks a texel uniformly over
// the whole WxH grid, then xi.y's alias/prob comparison redirects it to the
// weighted texel that bucket represents; the leftover precision in both
// (renormalized after the accept/reject split for xi.y, so it stays uniform
// regardless of branch) becomes the continuous sub-texel jitter, same role
// the old binary search's cdf-interval interpolation played
// @xi is a uniform random sample in [0,1)^2
// @rotation is the environment map's rotation in radians
// @pdf is the solid-angle pdf of the returned direction
public func environmentMapImportanceSample(
	handles: EnvironmentMapHandles,
	xi: f32v2,
	rotation: f32,
	out pdf: f32,
) -> f32v3 {
	const i32v2 size = textureSize(vkofTextures[handles.aliasProb], 0);
	const i32 width = size.x;
	const i32 height = size.y;
	const i32 n = width * height;

	const f32 nf = xi.x * f32(n);
	const i32 i = clamp(i32(nf), 0, n - 1);
	const f32 jitterX = nf - f32(i);

	const f32 probThreshold = (
		texelFetch(vkofTextures[handles.aliasProb], i32v2(i % width, i / width), 0).r
	);
	const bool accept = xi.y < probThreshold;
	const i32 finalIdx = (
		accept
			? i
			: i32(
				texelFetch(
					vkofTextures[handles.aliasIndex], i32v2(i % width, i / width), 0
				).r + 0.5f
			)
	);
	const f32 jitterY = (
		accept
			? xi.y / max(probThreshold, 1e-8f)
			: (xi.y - probThreshold) / max(1.0f - probThreshold, 1e-8f)
	);

	const i32 x0 = finalIdx % width;
	const i32 y0 = finalIdx / width;

	// -- convert to continuous uv coordinate
	const f32v2 uv = (
		f32v2(
			(f32(x0) + jitterX) / f32(width),
			(f32(y0) + jitterY) / f32(height)
		)
	);

	// -- convert uv to world-space direction and compute pdf
	const f32v3 dir = environmentMapUvToDir(uv, rotation);
	pdf = environmentMapPdf(handles, dir, rotation);
	return dir;
}

// filtered (bilinear) radiance lookup; filtered radiance against an
// exact-texel pdf is a standard, harmless approximation (pbrt does the same)
public func environmentMapRadiance(
	handles: EnvironmentMapHandles,
	dir: f32v3,
	rotation: f32,
) -> f32v3 {
	const f32v2 uv = environmentMapDirToUv(dir, rotation);
	return texture(vkofTextures[handles.radiance], uv).rgb;
}

