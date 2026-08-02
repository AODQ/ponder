import "util-material-openpbr-thinfilm";
import "util-material-openpbr-shared";

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
	const u32 i = thread * 8u;
	const u32 o = thread * 6u;

	OpenPbrMaterial mat;
	mat.thinFilmWeight = 1.0f;
	mat.thinFilmThickness = inBuf[i + 0u];
	mat.thinFilmIor = inBuf[i + 1u];
	mat.specularIor = inBuf[i + 2u];

	const f32 cosThetaI = inBuf[i + 3u];
	const f32v3 etaBase = f32v3(
		inBuf[i + 4u], inBuf[i + 4u] * 1.1f, inBuf[i + 4u] * 0.9f
	);
	const f32v3 kappaBase = f32v3(inBuf[i + 5u], 0.0f, inBuf[i + 5u] * 2.0f);

	const f32v3 film = openPbrThinfilmFresnel(mat, cosThetaI, etaBase, kappaBase);
	outBuf[o + 0u] = film.r;
	outBuf[o + 1u] = film.g;
	outBuf[o + 2u] = film.b;

	const f32v3 sens = utilEvaluateXyzSensitivity(inBuf[i + 6u], inBuf[i + 7u]);
	outBuf[o + 3u] = sens.x;
	outBuf[o + 4u] = sens.y;
	outBuf[o + 5u] = sens.z;
}
