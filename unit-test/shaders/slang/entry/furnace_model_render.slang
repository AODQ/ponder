import "environment";
import "util-random";
#include "util-nan-probe.slang"
import "util-raytrace";
import "util-mor-shared";
import "util-resolve-pc";

struct PC {
	GpuGlobalPc global;
	GpuFurnaceModelPC furnace;
};

[[vk::push_constant]] PC pc;

// PONDER_NAN_PROBES: temporary, for the DragonAttenuation firefly
// investigation -- see util-material-openpbr-transmission.glsl and
// util-raytrace.glsl's guarded debug blocks. remove alongside those once
// root-caused
#define PONDER_NAN_PROBES

// furnace-tests a real loaded glTF model through the exact production path
// tracer (utilTraceRay + utilIrradiancePropagationEntry, the same calls
// resolve.comp makes) instead of resolve.comp itself, which writes to an
// image vkof has no raw-float readback path for; this writes to a plain
// buffer like every other render-validation unit test (see
// cornell_openpbr_render.comp) so the result can be numerically checked for
// NaN/Inf and firefly pixels, not just eyeballed. seeding is a simple
// per-pixel-per-sample hash (fnPcgHash, same trick cornell's shader uses)
// rather than resolve.comp's bluenoise stream -- an unrelated variance-
// reduction detail, not part of what's being tested here

// RESOLVE_NO_PC: pulls in GpuGlobalPc/GpuResolveModelIndirect(Buffer) from
// resolve_pc.h without its own push_constant block, since this shader's
// output is a buffer (outVa) rather than resolve.comp's image + bluenoise
// pushconstant fields
#define RESOLVE_NO_PC

struct GpuFurnaceModelPC {
	u64 outVa;
	u32 width;
	u32 height;
	u32 sampleCount;
	u32 propagationDepth;
	u32 kullaContyEnergyHandle;
	u32 zeltnerLtcParamHandle;
	// decorrelates the per-pixel rng streams between otherwise-identical
	// dispatches, so a test can render independent realizations of the same
	// image (paired-noise variance estimation in test-environment-mis.cpp);
	// 0 reproduces the pre-salt stream exactly
	u32 seedSalt;
	u32 pad0;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	// (TODO REVIEW)
	utilRaytraceBind(pc.global.models);
	// (TODO REVIEW)
	const i32v2 coord = i32v2(gl_GlobalInvocationID.xy);
	if (coord.x >= i32(pc.furnace.width) || coord.y >= i32(pc.furnace.height)) {
		return;
	}
	const i32v2 screensize = i32v2(pc.furnace.width, pc.furnace.height);
	const u32 pixel = u32(coord.y) * pc.furnace.width + u32(coord.x);
	nanProbeInit(
		((GpuGlobalExtended *)(pc.global.extended))[0].nanProbeCounterVa,
		coord
	);

	const MaterialTableHandles materialTables = MaterialTableHandles(
		pc.furnace.kullaContyEnergyHandle,
		pc.furnace.zeltnerLtcParamHandle
	);

	// deterministic primary ray, no antialiasing jitter -- every sample
	// re-traces the exact same camera ray, only the bsdf-path rng differs
	const f32v2 uv = (f32v2(coord) + 0.5f) / f32v2(screensize);
	const f32v2 ndc = uv * 2.0f - 1.0f;
	const f32m4x4 invVP = inverse(pc.global.viewProj);
	const f32v4 worldFar = mul(invVP, f32v4(ndc, 1.0f, 1.0f));
	const f32v3 rayDir = (
		normalize(worldFar.xyz / worldFar.w - pc.global.cameraPos)
	);

	// utilIrradiancePropagationEntry owns the primary trace now, so this
	// walks the camera ray straight in rather than resolving the first hit
	// here and handing the entry a loaded surface -- the shape it had when
	// this file was written, and the reason it stopped compiling
	const GpuGlobalExtended unf = (
		((GpuGlobalExtended *)(pc.global.extended))[0]
	);
	UtilVdb vdb;
	vdb.isValid = false;

	f32v3 radianceSum = f32v3(0.0f);
	for (u32 s = 0u; s < pc.furnace.sampleCount; ++s) {
		// 7919 * salt keeps the salted streams disjoint from every
		// (pixel, s) pair of the unsalted stream: s < sampleCount << 7919
		// and 7919 < 9781, so no collision is reachable
		const u32 h0 = (
			fnPcgHash(pixel * 9781u + s + 1u + pc.furnace.seedSalt * 7919u)
		);
		const u32 h1 = fnPcgHash(h0);
		u64 state = (u64(h0) << 32) | u64(h1);

		const f32v4 irradiance = (
			utilIrradiancePropagationEntry(
				/*tables=*/materialTables,
				/*vdb=*/vdb,
				/*origin=*/pc.global.cameraPos,
				/*worldDir=*/rayDir,
				/*propagationDepth=*/pc.furnace.propagationDepth,
				/*state=*/state,
				/*unf=*/unf
			)
		);
		radianceSum += irradiance.rgb;
	}

	const f32v3 radiance = radianceSum / f32(pc.furnace.sampleCount);
	const u32 pixelCount = pc.furnace.width * pc.furnace.height;
	f32 * outBuf = (f32 *)(pc.furnace.outVa);
	outBuf[pixel] = radiance.r;
	outBuf[pixelCount + pixel] = radiance.g;
	outBuf[2u * pixelCount + pixel] = radiance.b;
}
