import "environment";
import "util-random";
import "util-raytrace";
import "util-mor-shared";
import "util-resolve-pc";

struct PC {
	GpuGlobalPc global;
	GpuSubsurfaceFurnacePC furnace;
};

[[vk::push_constant]] PC pc;

// furnace-tests a real closed-mesh scene with a subsurface material through
// the exact production path tracer (utilTraceRay + utilIrradiance
// PropagationEntry, the same calls resolve.comp makes) -- this is the only
// way to exercise utilSubsurfaceWalk / utilLoadSurfaceHit's mesh-hit
// unpacking for real, rather than re-testing the primitives already
// covered in isolation (test-subsurface-albedo.cpp, test-subsurface-walk
// .cpp, test-subsurface-entry-sampling.cpp). modeled on the disabled
// furnace_model_render.comp (test-furnace-model-render.cpp), updated to
// the current utilIrradiancePropagationEntry(tables, vdb, origin,
// worldDir, propagationDepth, state, unf) signature -- that file's own
// call predates the vdb/GpuGlobalExtended parameters and no longer
// compiles against current util-raytrace.glsl; unrelated to subsurface,
// not fixed here.

#define RESOLVE_NO_PC

struct GpuSubsurfaceFurnacePC {
	u64 outVa;
	u32 width;
	u32 height;
	u32 sampleCount;
	u32 propagationDepth;
	u32 kullaContyEnergyHandle;
	u32 zeltnerLtcParamHandle;
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

	const MaterialTableHandles materialTables = MaterialTableHandles(
		pc.furnace.kullaContyEnergyHandle,
		pc.furnace.zeltnerLtcParamHandle
	);

	const f32v2 uv = (f32v2(coord) + 0.5f) / f32v2(screensize);
	const f32v2 ndc = uv * 2.0f - 1.0f;
	const f32m4x4 invVP = inverse(pc.global.viewProj);
	const f32v4 worldFar = mul(invVP, f32v4(ndc, 1.0f, 1.0f));
	const f32v3 rayDir = (
		normalize(worldFar.xyz / worldFar.w - pc.global.cameraPos)
	);

	const GpuGlobalExtended unf = ((GpuGlobalExtended *)(pc.global.extended))[0];

	f32v3 radianceSum = f32v3(0.0f);
	for (u32 s = 0u; s < pc.furnace.sampleCount; ++s) {
		const u32 h0 = (
			fnPcgHash(pixel * 9781u + s + 1u + pc.furnace.seedSalt * 7919u)
		);
		const u32 h1 = fnPcgHash(h0);
		u64 state = (u64(h0) << 32) | u64(h1);

		UtilVdb vdb;
		vdb.isValid = false;

		const f32v4 irradiance = (
			utilIrradiancePropagationEntry(
				materialTables, vdb, pc.global.cameraPos, rayDir,
				pc.furnace.propagationDepth, state, unf
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
