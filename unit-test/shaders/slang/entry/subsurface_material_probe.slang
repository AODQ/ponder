import "environment";
import "util-random";
import "util-raytrace";
import "util-mor-shared";
import "util-resolve-pc";

struct PC {
	GpuGlobalPc global;
	GpuSubsurfaceProbePC probe;
};

[[vk::push_constant]] PC pc;

// DEBUG-only probe: traces the camera ray through utilTraceRay (same as
// production) and dumps whatever openPbrLoadMaterialDeriv/utilLoadSurfaceHit
// actually resolves for subsurfaceColor/subsurfaceRadius/subsurfaceWeight at
// the hit point, instead of running the real irradiance walk -- isolates
// whether the material override is reaching the shader at all versus a bug
// further down the pipeline.

#define RESOLVE_NO_PC

struct GpuSubsurfaceProbePC {
	u64 outVa;
	u32 width;
	u32 height;
	// 0 = subsurfaceColor, 1 = coatColor, 2 = subsurfaceRadiusScale
	u32 fieldSelect;
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
	if (coord.x >= i32(pc.probe.width) || coord.y >= i32(pc.probe.height)) {
		return;
	}
	const i32v2 screensize = i32v2(pc.probe.width, pc.probe.height);
	const u32 pixel = u32(coord.y) * pc.probe.width + u32(coord.x);

	const f32v2 uv = (f32v2(coord) + 0.5f) / f32v2(screensize);
	const f32v2 ndc = uv * 2.0f - 1.0f;
	const f32m4x4 invVP = inverse(pc.global.viewProj);
	const f32v4 worldFar = mul(invVP, f32v4(ndc, 1.0f, 1.0f));
	const f32v3 rayDir = (
		normalize(worldFar.xyz / worldFar.w - pc.global.cameraPos)
	);

	f32 * outBuf = (f32 *)(pc.probe.outVa);
	const u32 pixelCount = pc.probe.width * pc.probe.height;

	const RayQueryResult rq = (
		utilTraceRay(pc.global.cameraPos, rayDir, 999999.0f, false, true)
	);
	if (rq.dist <= 0.0f) {
		outBuf[pixel] = -1.0f;
		outBuf[pixelCount + pixel] = -1.0f;
		outBuf[2u * pixelCount + pixel] = -1.0f;
		return;
	}

	f32v3 itOri = pc.global.cameraPos;
	OpenPbrMaterial material;
	ShadingFrame itFrame;
	ShadingFrame itCoatFrame;
	f32v3 itNormalGeometrical;
	f32v3 itWi;
	utilLoadSurfaceHit(
		rq, itOri, rayDir,
		material, itFrame, itCoatFrame, itNormalGeometrical, itWi
	);

	f32v3 chosen = material.subsurfaceColor;
	if (pc.probe.fieldSelect == 1u) {
		chosen = material.coatColor;
	} else if (pc.probe.fieldSelect == 2u) {
		chosen = material.subsurfaceRadiusScale;
	}
	outBuf[pixel] = chosen.r;
	outBuf[pixelCount + pixel] = chosen.g;
	outBuf[2u * pixelCount + pixel] = chosen.b;

	// DEBUG: reads the exact same bytes via a raw-float reinterpretation
	// instead of the GpuMorMaterial struct type, bypassing whatever
	// buffer(decorated struct)->value-type-struct copy the compiler
	// generates for the .rgb field access above -- byte offset 304 is
	// subsurfaceColor's confirmed-correct offset (spirv-dis
	// OpMemberDecorate matches C++ offsetof exactly), float index 76 =
	// 304/4. single-material scene, materialIndex 0 assumed
	const GpuResolveModelIndirect modelDesc = (
		((GpuResolveModelIndirect *)(pc.global.models))[rq.modelDrawIndex]
	);
	f32 * rawMaterialFloats = (f32 *)(modelDesc.materials);
	outBuf[3u * pixelCount + pixel] = rawMaterialFloats[76];
	outBuf[4u * pixelCount + pixel] = rawMaterialFloats[77];
	outBuf[5u * pixelCount + pixel] = rawMaterialFloats[78];

	// DEBUG: direct per-field access on the buffer-indexed expression,
	// never materializing a whole GpuMorMaterial value-type local at all
	// (no "GpuMorMaterial mat = buf[idx];") -- tests whether the bug
	// is specifically in the bulk struct-to-struct copy
	GpuMorMaterial * matBuf = (GpuMorMaterial *)(modelDesc.materials);
	const f32v3 directField = matBuf[0].subsurfaceColor.rgb;
	outBuf[6u * pixelCount + pixel] = directField.x;
	outBuf[7u * pixelCount + pixel] = directField.y;
	outBuf[8u * pixelCount + pixel] = directField.z;

	// DEBUG: openPbrLoadMaterialDeriv called directly here, bypassing
	// utilLoadSurfaceHit's own `out OpenPbrMaterial material` parameter
	// entirely -- isolates whether the corruption is now in
	// openPbrLoadMaterialDeriv's own OpenPbrMaterial return (a
	// structurally similar large multi-vec3-field struct), or in how
	// utilLoadSurfaceHit's out-parameter copies it out afterward
	f32v3 dgn, dgcn;
	f32v4 dgt, dgct;
	const OpenPbrMaterial directMat = (
		openPbrLoadMaterialDeriv(
			matBuf, 0u,
			(GpuMorUvTransform *)(modelDesc.uvTransforms),
			f32v2(0.0f), f32v2(0.0f), f32v2(0.0f),
			dgn, dgt, dgcn, dgct
		)
	);
	outBuf[9u * pixelCount + pixel] = directMat.subsurfaceColor.x;
	outBuf[10u * pixelCount + pixel] = directMat.subsurfaceColor.y;
	outBuf[11u * pixelCount + pixel] = directMat.subsurfaceColor.z;
}
