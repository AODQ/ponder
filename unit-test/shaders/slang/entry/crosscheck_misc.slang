import "util-blackbody";
import "util-fog";
import "util-tonemap";
import "util-environment-map";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
};

[[vk::push_constant]] PC pc;

// the openpbr headers expect their includer to supply all of these
#ifndef f32
#define i32 i32
#define i32v2 i32v2
#define f32 f32
#define f32v2 f32v2
#define f32v3 f32v3
#define f32v4 f32v4
#define f32m44 f32m4x4
#define u32 u32
#define u32v2 u32v2
#define u32v3 u32v3
#define u32v4 u32v4
#define u64 u64
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32 thread = gl_GlobalInvocationID.x;
	if (thread >= pc.threadCount) { return; }
	f32 * inBuf = (f32 *)(pc.inVa);
	f32 * outBuf = (f32 *)(pc.outVa);
	const u32 i = thread * 6u;
	const u32 o = thread * 12u;

	const f32v3 bb = utilBlackbodyRadiance(inBuf[i + 0u]);
	outBuf[o + 0u] = bb.r;
	outBuf[o + 1u] = bb.g;
	outBuf[o + 2u] = bb.b;

	outBuf[o + 3u] = utilFogTransmittance(inBuf[i + 1u], inBuf[i + 2u]);

	const f32v3 aces = fnAces(f32v3(
		inBuf[i + 1u], inBuf[i + 2u], inBuf[i + 3u]
	));
	outBuf[o + 4u] = aces.r;
	outBuf[o + 5u] = aces.g;
	outBuf[o + 6u] = aces.b;

	const f32v3 dir = normalize(f32v3(
		inBuf[i + 3u], inBuf[i + 4u], inBuf[i + 5u] + 0.01f
	));
	const f32v2 uv = environmentMapDirToUv(dir, inBuf[i + 0u] * 0.001f);
	outBuf[o + 7u] = uv.x;
	outBuf[o + 8u] = uv.y;

	// round-trips back through the inverse; a sign or wrap error shows here
	const f32v3 back = environmentMapUvToDir(uv, inBuf[i + 0u] * 0.001f);
	outBuf[o + 9u] = back.x;
	outBuf[o + 10u] = back.y;
	outBuf[o + 11u] = back.z;
}
