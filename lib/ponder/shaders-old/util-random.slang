module "util-random";
__exported import "util-types";

public static const f32 skPi = 3.14159265358979323846;
public static const f32 skTau = 6.28318530717958647692;

// one-shot hash, used only to seed the pcg32 streams
public func fnPcgHash(s0: u32) -> u32 {
	var s = s0 * 747796405u + 2891336453u;
	s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
	return (s >> 22u) ^ s;
}

public func fnUintToUniform(s: u32) -> f32 {
	return asfloat((s >> 9u) | 0x3F800000u) - 1.0f;
}

// oneill pcg32. 64-bit lcg state, xorshift-rotate output. full 2^64 period
public func fnPcg32Output(state: u64) -> u32 {
	const u32 xorshifted = u32(((state >> 18u) ^ state) >> 27u);
	const u32 rot = u32(state >> 59u);
	return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

public func fnSampleUniform(inout state: u64) -> f32 {
	const u32 outValue = fnPcg32Output(state);
	state = state * 6364136223846793005ul + 1442695040888963407ul;
	return fnUintToUniform(outValue);
}

public func fnSampleUniform2(inout state: u64) -> f32v2 {
	return f32v2(fnSampleUniform(state), fnSampleUniform(state));
}

// uniform direction on a unit sphere; pdf = 1/4*pi
public func fnSampleUniformSphere(u: f32v2) -> f32v3 {
	const f32 z = 1.0 - 2.0 * u.x;
	const f32 r = sqrt(max(0.0, 1.0 - z * z));
	const f32 phi = skTau * u.y;
	return f32v3(r * cos(phi), r * sin(phi), z);
}

public func fnSampleBluenoise(coord: u32v2, textureHandle: u32) -> f32 {
	const f32v2 uv = (f32v2(coord % u32v2(128)) + 0.5f) / 128.0f;
	return (
		vkofTextures[
			NonUniformResourceIndex(textureHandle)
		].SampleLevel(uv, 0.0f).r
	);
}

// seeds two independent pcg32 streams; state/state2 are for further draws,
// seed/seed2 are the first sample from each. bluenoise blends into the
// first scalar sample only; cranley-patterson rotation.
//
// deviates from the glsl by taking the invocation id: slang has no free
// gl_GlobalInvocationID, it only reaches an entry point as a semantic
public func utilGenerateSeeds(
	out state: u64,
	out state2: u64,
	out seed: f32,
	out seed2: f32v2,
	invocationId: u32v2,
	frameIndex: u32,
	salt: u32,
	bluenoiseTextureHandle: u32,
) {
	const u32 h0 = fnPcgHash(frameIndex + salt * 0x9E3779B9u);
	const u32 h1 = fnPcgHash(invocationId.y + h0);
	const u32 h2 = fnPcgHash(invocationId.x + h1);
	const u32 h3 = fnPcgHash(h2);
	const u32 h4 = fnPcgHash(h3);

	state = (u64(h2) << 32) | u64(h3);
	state2 = (u64(h3) << 32) | u64(h4);

	const f32 bluenoise = (
		fnSampleBluenoise(invocationId, bluenoiseTextureHandle)
	);
	seed = frac(bluenoise + fnSampleUniform(state));
	seed2 = fnSampleUniform2(state2);
}
