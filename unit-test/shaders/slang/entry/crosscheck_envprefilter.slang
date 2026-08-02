import "util-shading-frame";
import "util-env-prefilter";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
};

[[vk::push_constant]] PC pc;

// util-shading-frame pulls in util-random, which reaches for the bluenoise
// texture set; this shader never calls that path

// all four of util-env-prefilter: the van der corput bit reversal, the
// hammersley pair, karis's ggx half-vector, and the krivanek-colbert lod

// in:  index, sampleCount, alpha, nor(3), pdf, maxMipLevel
// out: vdc, hammersley(2), h(3), lod
[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 thread = gl_GlobalInvocationID.x;
	if (thread >= pc.threadCount) { return; }

	f32 * inBuf = (f32 *)(pc.inVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 i = thread * 8u;
	const u32 o = thread * 7u;

	const u32 index = asuint(inBuf[i + 0u]);
	const u32 sampleCount = u32(inBuf[i + 1u]);
	const f32 alpha = inBuf[i + 2u];
	const f32v3 nor = normalize(
		f32v3(inBuf[i + 3u], inBuf[i + 4u], inBuf[i + 5u])
	);
	const f32 pdf = inBuf[i + 6u];
	const f32 maxMipLevel = inBuf[i + 7u];

	outBuf[o + 0u] = envPrefilterRadicalInverseVdc(index);

	const f32v2 xi = envPrefilterHammersley(index, sampleCount);
	outBuf[o + 1u] = xi.x;
	outBuf[o + 2u] = xi.y;

	const f32v3 h = envPrefilterImportanceSampleH(xi, alpha, nor);
	outBuf[o + 3u] = h.x;
	outBuf[o + 4u] = h.y;
	outBuf[o + 5u] = h.z;

	outBuf[o + 6u] = (
		envPrefilterComputeLod(h, pdf, maxMipLevel, sampleCount)
	);
}
