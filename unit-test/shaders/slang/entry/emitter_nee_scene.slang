import "environment";
import "util-random";
#include "util-nan-probe.slang"
import "util-raytrace";
import "util-vdb";
import "util-mor-shared";
import "util-resolve-pc";

struct PC {
	GpuGlobalPc global;
	GpuEmitterScenePC scene;
};

[[vk::push_constant]] PC pc;

// tier 3 of plan-nee.md: renders a scene through the *production* path
// tracer -- utilIrradiancePropagationEntry, the same call resolve.comp makes
// -- so the emissive-mesh nee integration is exercised end to end: path-state
// threading, the delta exclusion, the backward hit lookup against real traced
// geometry.
//
// the one thing it deliberately does not reuse is the viewer's accumulation:
// pt-accumulate.comp's average is over samples, and comparing two configs
// that differ in how often a path carries light means comparing two different
// normalizations. this averages radianceSum / sampleCount flat, so an
// nee-on/nee-off comparison measures the estimator and nothing else.

struct GpuEmitterScenePC {
	u64 outVa;
	u32 width;
	u32 height;
	u32 sampleCount;
	u32 propagationDepth;
	u32 kullaContyEnergyHandle;
	u32 zeltnerLtcParamHandle;
	// decorrelates otherwise-identical dispatches so the same config can be
	// rendered as independent realizations (paired-noise error bars)
	u32 seedSalt;
	u32 pad0;
};

#define RESOLVE_NO_PC

[shader("compute")]
[numthreads(8, 8, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	// (TODO REVIEW)
	utilRaytraceBind(pc.global.models);
	// (TODO REVIEW)
	const i32v2 coord = i32v2(gl_GlobalInvocationID.xy);
	if (coord.x >= i32(pc.scene.width) || coord.y >= i32(pc.scene.height)) {
		return;
	}
	const i32v2 screensize = i32v2(pc.scene.width, pc.scene.height);
	const u32 pixel = u32(coord.y) * pc.scene.width + u32(coord.x);

	const GpuGlobalExtended unf = (
		((GpuGlobalExtended *)(pc.global.extended))[0]
	);
	nanProbeInit(unf.nanProbeCounterVa, coord);

	const MaterialTableHandles materialTables = MaterialTableHandles(
		pc.scene.kullaContyEnergyHandle,
		pc.scene.zeltnerLtcParamHandle
	);

	// no antialiasing jitter: every sample retraces the identical camera ray,
	// so the only difference between two configs is the path rng
	const f32v2 uv = (f32v2(coord) + 0.5f) / f32v2(screensize);
	const f32v2 ndc = uv * 2.0f - 1.0f;
	const f32m4x4 invVP = inverse(pc.global.viewProj);
	const f32v4 worldFar = mul(invVP, f32v4(ndc, 1.0f, 1.0f));
	const f32v3 rayDir = (
		normalize(worldFar.xyz / worldFar.w - pc.global.cameraPos)
	);

	UtilVdb vdb;
	vdb.isValid = false;

	f32v3 radianceSum = f32v3(0.0f);
	for (u32 s = 0u; s < pc.scene.sampleCount; ++s) {
		const u32 h0 = (
			fnPcgHash(pixel * 9781u + s + 1u + pc.scene.seedSalt * 7919u)
		);
		const u32 h1 = fnPcgHash(h0);
		u64 state = (u64(h0) << 32) | u64(h1);

		const f32v4 irradiance = (
			utilIrradiancePropagationEntry(
				/*tables=*/materialTables,
				/*vdb=*/vdb,
				/*origin=*/pc.global.cameraPos,
				/*worldDir=*/rayDir,
				/*propagationDepth=*/pc.scene.propagationDepth,
				/*state=*/state,
				/*unf=*/unf
			)
		);
		// flat average over every sample, including the ones that carried no
		// light -- see the header
		radianceSum += irradiance.rgb;
	}

	const f32v3 radiance = radianceSum / f32(pc.scene.sampleCount);
	const u32 pixelCount = pc.scene.width * pc.scene.height;
	f32 * outBuf = (f32 *)(pc.scene.outVa);
	outBuf[pixel] = radiance.r;
	outBuf[pixelCount + pixel] = radiance.g;
	outBuf[2u * pixelCount + pixel] = radiance.b;
}
