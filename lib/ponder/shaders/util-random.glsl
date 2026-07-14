#ifndef UTIL_RANDOM_GLSL
#define UTIL_RANDOM_GLSL

#ifndef f32
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

// one-shot hash, used only to seed the pcg32 streams
u32 fnPcgHash(u32 s) {
	s = s * 747796405u + 2891336453u;
	s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
	return (s >> 22u) ^ s;
}

f32 fnUintToUniform(const u32 s) {
	return uintBitsToFloat((s >> 9u) | 0x3F800000u) - 1.0f;
}

// oneill pcg32. 64-bit lcg state, xorshift-rotate output. full 2^64 period
u32 fnPcg32Output(const u64 state) {
	const u32 xorshifted = u32(((state >> 18u) ^ state) >> 27u);
	const u32 rot = u32(state >> 59u);
	return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

f32 fnSampleUniform(inout u64 state) {
	const u32 out_ = fnPcg32Output(state);
	state = state * 6364136223846793005ul + 1442695040888963407ul;
	return fnUintToUniform(out_);
}

f32v2 fnSampleUniform2(inout u64 state) {
	return f32v2(fnSampleUniform(state), fnSampleUniform(state));
}

f32 fnSampleBluenoise(const u32v2 coord, const u32 textureHandle) {
	const f32v2 uv = (f32v2(coord % u32v2(128)) + 0.5f) / 128.0f;
	return texture(vkofTextures[nonuniformEXT(textureHandle)], uv).r;
}

// seeds two independent pcg32 streams; state/state2 are for further draws,
// seed/seed2 are the first sample from each. bluenoise blends into the
// first scalar sample only; cranley-patterson rotation.
void utilGenerateSeeds(
	out u64 state,
	out u64 state2,
	out f32 seed,
	out f32v2 seed2,
	const u32 frameIndex,
	const u32 salt,
	const u32 bluenoiseTextureHandle
) {
	const u32 h0 = fnPcgHash(frameIndex + salt * 0x9E3779B9u);
	const u32 h1 = fnPcgHash(gl_GlobalInvocationID.y + h0);
	const u32 h2 = fnPcgHash(gl_GlobalInvocationID.x + h1);
	const u32 h3 = fnPcgHash(h2);
	const u32 h4 = fnPcgHash(h3);

	state = (u64(h2) << 32) | u64(h3);
	state2 = (u64(h3) << 32) | u64(h4);

	const f32 bluenoise = fnSampleBluenoise(
		u32v2(gl_GlobalInvocationID.xy), bluenoiseTextureHandle
	);
	seed = fract(bluenoise + fnSampleUniform(state));
	seed2 = fnSampleUniform2(state2);
}

#endif // UTIL_RANDOM_GLSL
