import "util-environment-map";
import "util-shading-frame";
import "util-env-prefilter";
import "util-env-prefilter-pc";

struct PC {
	u32v4 rootPad[8];
	GpuEnvPrefilterPc prefilter;
};

[[vk::push_constant]] PC pc;

// cosine-hemisphere convolution; cosine-weighted sampling's pdf (cosTheta/pi)
// cancels both the brdf's 1/pi and the integral's cosTheta term exactly, so
// the plain sample mean is already "albedo * all(mean == diffuse) reflected
// radiance", no extra normalization needed at the shading call site
[shader("compute")]
[numthreads(8, 8, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const i32v2 coord = i32v2(gl_GlobalInvocationID.xy);
	if (
		coord.x >= i32(pc.prefilter.dstWidth)
		|| coord.y >= i32(pc.prefilter.dstHeight)
	) {
		return;
	}
	const f32v2 uv = (
		(f32v2(coord) + 0.5f)
		/ f32v2(pc.prefilter.dstWidth, pc.prefilter.dstHeight)
	);
	const f32v3 n = environmentMapUvToDir(uv, 0.0f);

	// a hemisphere-wide cosine lobe is wider than even the roughest
	// specular mip, so always read the source's coarsest mip -- see
	// env-prefilter-specular.comp's srcLod comment for why point-sampling
	// the full-res source doesn't converge
	const f32 srcLod = (
		f32(textureQueryLevels(vkofTextures[pc.prefilter.srcRadianceHandle]) - 1)
	);

	f32v3 sum = f32v3(0.0f);
	const u32 sampleCount = pc.prefilter.sampleCount;
	for (u32 i = 0u; i < sampleCount; ++i) {
		const f32v2 xi = envPrefilterHammersley(i, sampleCount);
		const f32v3 l = utilCosineHemisphereSampleWo(n, xi);
		const f32v2 srcUv = environmentMapDirToUv(l, 0.0f);
		sum += (
			textureLod(vkofTextures[pc.prefilter.srcRadianceHandle], srcUv, srcLod).rgb
		);
	}
	const f32v3 result = sum / f32(sampleCount);

	imageStore(vkofImages[pc.prefilter.dstStorageHandle], coord, f32v4(result, 1.0f));
}
