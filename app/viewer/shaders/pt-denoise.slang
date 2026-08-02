import "util-tonemap";
import "util-global-pc";
import "util-pt-denoise-pc";

struct PC {
	GpuGlobalPc global;
	GpuDenoisePC denoise;
};

[[vk::push_constant]] PC pc;

// variance-adaptive a-trous edge-avoiding spatial filter, ported from
// ~/repo/demo/shaders/filter.comp. filters albedo-demodulated lighting
// (not raw color) so texture detail survives; the final pass remultiplies
// the center pixel's own albedo back in. no temporal reprojection needed --
// pt-accumulate.comp's running mean is already the correct unbiased history
// since ponder only accumulates with a static camera

static const f32 kAlbedoEpsilon = 0.05f;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const GpuGlobalExtended unf = (
		((GpuGlobalExtended *)(pc.global.extended))[0]
	);
	const i32v2 coord = i32v2(gl_GlobalInvocationID.xy);
	const i32v2 screensize = i32v2(unf.renderWidth, unf.renderHeight);
	if (coord.x >= screensize.x || coord.y >= screensize.y) { return; }

	const f32v4 posC = (
		imageLoad(vkofImages[pc.denoise.gbufferPosHandle], coord)
	);
	const f32v4 norC = (
		imageLoad(vkofImages[pc.denoise.gbufferNorHandle], coord)
	);
	const f32v3 albedoC = max(
		imageLoad(vkofImages[pc.denoise.gbufferAlbedoHandle], coord).rgb,
		f32v3(kAlbedoEpsilon)
	);
	const f32v4 colC = imageLoad(vkofImages[pc.denoise.inputHandle], coord);
	const f32v3 lightC = (
		pc.denoise.isFirstPass != 0u ? colC.rgb / albedoC : colC.rgb
	);
	// count is the real sample count pt-accumulate.comp tracked, so its
	// reciprocal is a directly-founded variance proxy
	const f32 count = (
		imageLoad(vkofImages[pc.denoise.accumHandle], coord).a
	);
	const f32 variance = 1.0f / max(count, 1.0f);

	const f32 sigmaL = 4.0f;
	const f32 sigmaP = pc.denoise.positionSigma;
	// loose enough that a smoothly curved surface still blends -- 128 only
	// let near-identical normals through, starving curved geometry of any
	// real averaging while flatter areas over-smoothed
	const f32 sigmaN = 32.0f;

	const i32 step = i32(pc.denoise.stepSize);
	// full 3x3, not a 5-tap cross -- a cross only samples 4 directions per
	// pass, which isn't enough real averaging to knock down noise
	const i32v2 offsets[9] = {
		i32v2(0, 0),
		i32v2(1, 0), i32v2(-1, 0), i32v2(0, 1), i32v2(0, -1),
		i32v2(1, 1), i32v2(1, -1), i32v2(-1, 1), i32v2(-1, -1)
	};

	f32v3 sumLight = f32v3(0.0f);
	f32 sumWeight = 0.0f;
	for (i32 i = 0; i < 9; ++i) {
		const i32v2 npx = coord + offsets[i] * step;
		if (
			any((npx < i32v2(0)))
			|| any((npx >= screensize))
		) {
			continue;
		}
		const f32v4 posQ = (
			imageLoad(vkofImages[pc.denoise.gbufferPosHandle], npx)
		);
		// never blend a background/geometry mismatch
		if (posQ.w != posC.w) { continue; }
		const f32v4 norQ = (
			imageLoad(vkofImages[pc.denoise.gbufferNorHandle], npx)
		);
		const f32v4 colQ = imageLoad(vkofImages[pc.denoise.inputHandle], npx);
		f32v3 lightQ = colQ.rgb;
		if (pc.denoise.isFirstPass != 0u) {
			const f32v3 albedoQ = max(
				imageLoad(vkofImages[pc.denoise.gbufferAlbedoHandle], npx).rgb,
				f32v3(kAlbedoEpsilon)
			);
			lightQ /= albedoQ;
		}

		const f32 lumC = dot(lightC, f32v3(0.2126f, 0.7152f, 0.0722f));
		const f32 lumQ = dot(lightQ, f32v3(0.2126f, 0.7152f, 0.0722f));

		const bool isSurface = posC.w > 0.5f;
		const f32 wN = (
			isSurface
				? pow(max(dot(norC.xyz, norQ.xyz), 0.0f), sigmaN)
				: 1.0f
		);
		const f32 wP = (
			isSurface
				? exp(-distance(posC.xyz, posQ.xyz) / (sigmaP + 1e-4f))
				: 1.0f
		);
		// relative, not absolute: an HDR path tracer's luminance spans many
		// orders of magnitude, so a fixed-unit sigma is either too tight in
		// dim regions or too loose near bright ones
		const f32 wL = (
			exp(
				-abs(lumC - lumQ)
				/ (sigmaL * sqrt(variance) * max(lumC, 1e-2f) + 1e-4f)
			)
		);

		const f32 w = wN * wP * wL;
		sumLight += lightQ * w;
		sumWeight += w;
	}

	const f32v3 outLight = sumLight / max(sumWeight, 1e-4f);

	if (pc.denoise.isFinalPass != 0u) {
		const f32v3 finalLinear = outLight * albedoC;
		imageStore(
			vkofImages[pc.denoise.outputHandle], coord,
			f32v4(fnAces(finalLinear * pc.global.exposure), 1.0f)
		);
	} else {
		imageStore(
			vkofImages[pc.denoise.outputHandle], coord, f32v4(outLight, 1.0f)
		);
	}
}
