import "util-environment-map";
import "util-material-openpbr";
import "util-env-prefilter";
import "util-env-prefilter-pc";

struct PC {
	u32v4 rootPad[8];
	GpuEnvPrefilterPc prefilter;
};

[[vk::push_constant]] PC pc;

// karis' split-sum prefilter : importance-sample the ggx half-vector
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

	// sampling a raw, full-res source point-hits or misses a small bright
	// feature (e.g. a sun disc) with no middle ground -- no sample count
	// converges that. per-sample pdf-based mip selection (krivanek & colbert
	// 2008, see envPrefilterComputeLod) reads an already-averaged, coarser
	// source mip when a sample represents a wide solid angle, matching
	// materialx's mx_generate_prefilter_env.glsl
	const f32 maxMipLevel = f32(pc.prefilter.srcRadianceMipCount - 1u);
	const f32 alpha = pc.prefilter.roughness;

	f32v3 sum = f32v3(0.0f);
	f32 weightSum = 0.0f;
	const u32 sampleCount = pc.prefilter.sampleCount;
	for (u32 i = 0u; i < sampleCount; ++i) {
		const f32v2 xi = envPrefilterHammersley(i, sampleCount);
		const f32v3 h = envPrefilterImportanceSampleH(xi, alpha, n);
		const f32v3 wo = reflect(-n, h);
		const f32 dotNorWo = max(dot(n, wo), 0.0f);
		if (dotNorWo > 0.0f) {
			const f32 dotNorH = max(dot(n, h), 0.0f);
			const f32 d = utilMicrofacetGgxDistribution(dotNorH, alpha);
			const f32 g1 = utilMicrofacetSmithG1(1.0f, alpha);
			const f32 pdf = d * g1 / 4.0f;
			const f32 lod = (
				envPrefilterComputeLod(wo, pdf, maxMipLevel, sampleCount)
			);
			const f32v2 srcUv = environmentMapDirToUv(wo, 0.0f);
			sum += (
				textureLod(vkofTextures[pc.prefilter.srcRadianceHandle], srcUv, lod).rgb
				* dotNorWo
			);
			weightSum += dotNorWo;
		}
	}
	const f32v3 result = weightSum > 0.0f ? sum / weightSum : f32v3(0.0f);

	imageStore(vkofImages[pc.prefilter.dstStorageHandle], coord, f32v4(result, 1.0f));
}
