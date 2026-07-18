#ifndef UTIL_ENVIRONMENT_MAP_GLSL
#define UTIL_ENVIRONMENT_MAP_GLSL

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

#ifndef PI
#define PI  3.14159265358979323846
#endif

// HDR environment map precomputed importance-sampling tables.
// radiance is r32g32b32a32 float WxH
// pdf is r32 float WxH
// conditionalCdf is r32 float (W+1)xH (one monotonic cdf row per texel row)
// marginalCdf is r32 float (H+1)x1
struct EnvironmentMapHandles {
	u32 radiance;
	u32 pdf;
	u32 conditionalCdf;
	u32 marginalCdf;
};

layout(buffer_reference, scalar) buffer EnvironmentMapHandlesBuffer {
	EnvironmentMapHandles data;
};

// world-space direction to equirect
f32v2 environmentMapDirToUv(const f32v3 dir, const f32 rotation) {
	const f32 theta = acos(clamp(dir.y, -1.0f, 1.0f));
	const f32 phi = atan(dir.z, dir.x);
	const f32 u = fract(0.5f + phi / (2.0f * PI) + rotation / (2.0f * PI));
	const f32 v = theta / PI;
	return f32v2(u, v);
}

// equirect uv to world-space direction; inverse of environmentMapDirToUv
f32v3 environmentMapUvToDir(const f32v2 uv, const f32 rotation) {
	const f32 theta = uv.y * PI;
	const f32 phi = (uv.x - 0.5f - rotation / (2.0f * PI)) * 2.0f * PI;
	const f32 sinTheta = sin(theta);
	const f32 cosTheta = cos(theta);
	return f32v3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
}

// binary-searches a monotonic cdf texture row for the cell containing xi.
// returns the index such cdf[i] <= xi < cdf[i+1]
// @row is the row of the cdf texture to search
// @count is the number of texels in that row to search
// @t is the fractional position of xi within the interval [cdf[i], cdf[i+1]]
// @xi is the random sample in [0,1) to invert through the cdf
i32 environmentMapCdfSearch(
	const u32 cdfHandle,
	const i32 row,
	const i32 count,
	const f32 xi,
	out f32 outFraction
) {
	i32 lo = 0;
	i32 hi = count;
	// -- search for the largest index lo such that cdf[lo] <= xi
	while (lo + 1 < hi) {
		const i32 mid = (lo + hi) / 2;
		const f32 v = texelFetch(vkofTextures[cdfHandle], ivec2(mid, row), 0).r;
		if (v <= xi) {
			lo = mid;
		} else {
			hi = mid;
		}
	}
	// -- compute fractional position of xi within the interval
	const f32 cdfLo = texelFetch(vkofTextures[cdfHandle], ivec2(lo, row), 0).r;
	const f32 cdfHi = (
		texelFetch(vkofTextures[cdfHandle], ivec2(lo + 1, row), 0).r
	);
	// linearly interpolate
	const f32 span = cdfHi - cdfLo;
	outFraction = span > 0.0f ? (xi - cdfLo) / span : 0.0f;
	return lo;
}

// solid-angle pdf of environment map's importance distribution at 'dir'
f32 environmentMapPdf(
	const EnvironmentMapHandles handles, const f32v3 dir, const f32 rotation
) {
	const f32v2 uv = environmentMapDirToUv(dir, rotation);
	const f32 sinTheta = sin(uv.y * PI);
	if (sinTheta <= 1e-6f) {
		return 0.0f;
	}
	// uv to discrete texel coordinate
	const i32v2 pdfSize = textureSize(vkofTextures[handles.pdf], 0);
	const i32v2 texel = ivec2(
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
// precomputed distribution.
// marginal cdf picks the row, the row's conditional cdf picks the column.
// pdf simply calls the environmentMapPdf directly
// @xi is a uniform random sample in [0,1)^2
// @rotation is the environment map's rotation in radians
// @pdf is the solid-angle pdf of the returned direction
f32v3 environmentMapImportanceSample(
	const EnvironmentMapHandles handles,
	const f32v2 xi,
	const f32 rotation,
	out f32 pdf
) {
	const i32v2 marginalSize = textureSize(vkofTextures[handles.marginalCdf], 0);
	const i32 height = marginalSize.x - 1;

	// -- compute the row index y0 and fractional position cdfFraction within row
	f32 cdfFractionV;
	const i32 y0 = (
		environmentMapCdfSearch(
			handles.marginalCdf,
			/*row=*/0,
			/*count=*/height,
			/*xi=*/xi.y,
			/*outFraction=*/cdfFractionV
		)
	);


	// -- compute the column index x0 and fractional position tu within column
	const i32v2 conditionalSize = (
		textureSize(vkofTextures[handles.conditionalCdf], 0)
	);
	const i32 width = conditionalSize.x - 1;
	f32 cdfFractionU;
	const i32 x0 = (
		environmentMapCdfSearch(
			handles.conditionalCdf,
			/*row=*/y0,
			/*count=*/width,
			/*xi=*/xi.x,
			/*outFraction=*/cdfFractionU
		)
	);

	// -- convert to continuous uv coordinate
	const f32v2 uv = (
		f32v2(
			(f32(x0) + cdfFractionU) / f32(width),
			(f32(y0) + cdfFractionV) / f32(height)
		)
	);

	// -- convert uv to world-space direction and compute pdf
	const f32v3 dir = environmentMapUvToDir(uv, rotation);
	pdf = environmentMapPdf(handles, dir, rotation);
	return dir;
}

// filtered (bilinear) radiance lookup; filtered radiance against an
// exact-texel pdf is a standard, harmless approximation (pbrt does the same)
f32v3 environmentMapRadiance(
	const EnvironmentMapHandles handles, const f32v3 dir, const f32 rotation
) {
	const f32v2 uv = environmentMapDirToUv(dir, rotation);
	return texture(vkofTextures[handles.radiance], uv).rgb;
}

#endif // UTIL_ENVIRONMENT_MAP_GLSL
