import "environment";
import "util-random";
#include "util-nan-probe.slang"
import "util-raytrace";
import "util-vdb";
import "util-mor-shared";
import "util-resolve-pc";
import "util-global-pc";

struct PC {
	GpuGlobalPc global;
	GpuResolvePC resolve;
};

[[vk::push_constant]] PC pc;

// path-traced resolve pass: reconstruct the primary camera ray, trace it,
// load the hit's OpenPBR material, and propagate irradiance through the
// scene (util-raytrace.glsl). writes raw (un-tonemapped) radiance;
// pt-accumulate.comp owns temporal accumulation, exposure, and tonemap

// enables lib/ponder's util-nan-probe.glsl NAN_CHECK* call sites (see
// util-material-openpbr.glsl); must be defined before the first #include
// that reaches util-nan-probe.glsl, since its macros are picked at that
// file's first inclusion. remove this line to strip every probe back to a
// no-op without touching lib/ponder at all
#define PONDER_NAN_PROBES

// (TODO REVIEW)
// rescales by luminance ratio rather than clamping per channel, so a clamped
// sample keeps its hue instead of shifting toward white. see
// GpuGlobalExtended.fireflyClampLuminance
func fnFireflyClamp(
	rgb: f32v3,
	limit: f32,
) -> f32v3 {
	const f32 luminance = dot(rgb, f32v3(0.2126f, 0.7152f, 0.0722f));
	if (limit <= 0.0f || luminance <= limit) {
		return rgb;
	}
	return rgb * (limit / luminance);
}
// (TODO REVIEW)

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	// (TODO REVIEW)
	utilRaytraceBind(pc.global.models);
	// (TODO REVIEW)
	const GpuGlobalExtended unf = (
		((GpuGlobalExtended *)(pc.global.extended))[0]
	);
	const i32v2 coord = (
		i32v2(gl_GlobalInvocationID.xy)
		+ i32v2(pc.resolve.tileOriginX, pc.resolve.tileOriginY)
	);
	const i32v2 screensize = i32v2(unf.renderWidth, unf.renderHeight);
	if (coord.x >= screensize.x || coord.y >= screensize.y) { return; }
	nanProbeInit(unf.nanProbeCounterVa, coord);

	u64 state;
	u64 state2;
	f32 seed;
	f32v2 seed2;
	{
		const u32 bluenoiseTextureHandle = (
			((u32 *)(pc.resolve.bluenoiseVa))[
				pc.resolve.frameIndex % pc.resolve.bluenoiseCount
			]
		);
		// utilGenerateSeeds derives its pixel identity from the tile-local
		// gl_GlobalInvocationID, which restarts at (0, 0) for every tiled
		// dispatch -- the packed tile origin as salt keeps streams distinct
		// across tiles, and stays 0 for a full-screen dispatch so untiled
		// frames seed exactly as before. its bluenoise lookup needs no such
		// fix: origins are multiples of the texture's 128px wrap period
		utilGenerateSeeds(
			state, state2, seed, seed2,
			/*invocationId=*/ u32v2(gl_GlobalInvocationID.xy),
			/*frameIndex=*/ pc.resolve.frameIndex,
			/*salt=*/ (pc.resolve.tileOriginY << 16) | pc.resolve.tileOriginX,
			bluenoiseTextureHandle
		);
	}

	// -- camera ray: reconstruct the primary ray; antialiasing jitters the
	// subpixel position per sample, otherwise every sample goes through the
	// exact pixel center
	// (TODO REVIEW)
	// a View frame is never averaged with anything, so jitter there buys no
	// antialiasing and only adds shimmer while the camera drags
	const bool isViewFrame = unf.frameMode == 0;
	const f32v2 jitter = (
		(unf.antialias != 0u && !isViewFrame) ? seed2 : f32v2(0.5f)
	);
	// (TODO REVIEW)
	const f32v2 uv = (f32v2(coord) + jitter) / f32v2(screensize);
	const f32v2 ndc = uv * 2.0f - 1.0f;
	const f32m4x4 invVP = inverse(pc.global.viewProj);
	const f32v4 worldFar = mul(invVP, f32v4(ndc, 1.0f, 1.0f));
	const f32v3 rayDir = normalize(worldFar.xyz / worldFar.w - pc.global.cameraPos);

	const MaterialTableHandles materialTables = (
		MaterialTableHandles(
			pc.resolve.kullaContyEnergyHandle,
			pc.resolve.zeltnerLtcParamHandle
		)
	);

	// FRAME_MODE_VIEW: camera is actively moving -- a 1spp env-lit preview.
	// the cpu sets envMapOnlyDepth to 2 for these frames, so depth 3 is the
	// whole path: primary ray, one real reflection/refraction segment (both
	// surfaces doing their env nee), then the environment. that second trace
	// is what lets a refracted ray actually leave the far side of a glass
	// object instead of dying inside it. see GpuGlobalExtended.frameMode's
	// comment for the view/clear/pathtrace sequence
	// (TODO REVIEW)
	if (isViewFrame) {
		// a black environment lights nothing, so the preview would be a black
		// screen -- fall back to the unlit base color that preview used before
		if (unf.envMode == ENV_MODE_BLACK) {
			const RayQueryResult hit = (
				utilTraceRay(
					pc.global.cameraPos,
					rayDir,
					999999.0f,
					/*isShadowRay=*/false,
					/*testAlphaCutoff=*/true
				)
			);
			f32v3 color = f32v3(0.0f);
			f32 alpha = 0.0f;
			if (hit.dist > 0.0f) {
				UtilMeshAttributeDataFromIndices rqData = (
					utilMeshAttributeDataFromIndices(
						pc.global.models,
						hit.modelDrawIndex,
						hit.primitiveIndex,
						hit.barycentric,
						/*unpackUvDerivatives=*/false
					)
				);
				f32v3 geometryNormal, geometryCoatNormal;
				f32v4 geometryTangent, geometryCoatTangent;
				const OpenPbrMaterial mat = (
					openPbrLoadMaterialDeriv(
						rqData.materialBuf,
						rqData.materialIndex,
						rqData.uvTransforms,
						rqData.uv,
						f32v2(0.0f),
						f32v2(0.0f),
						geometryNormal,
						geometryTangent,
						geometryCoatNormal,
						geometryCoatTangent
					)
				);
				color = mat.baseColor;
				alpha = 1.0f;
			}
			imageStore(
				vkofImages[pc.resolve.outputImageHandle], coord,
				f32v4(color, alpha)
			);
			return;
		}

		// no vdb bound: a delta-tracking walk is the one thing here that
		// isn't bounded by the two-ray budget, and the preview is about
		// surfaces reading right while the camera moves
		UtilVdb viewVdb;
		viewVdb.isValid = false;
		const f32v4 viewIrradiance = (
			utilIrradiancePropagationEntry(
				/*tables=*/materialTables,
				/*vdb=*/viewVdb,
				/*origin=*/pc.global.cameraPos,
				/*wi=*/rayDir,
				/*propagationDepth=*/3u,
				/*state=*/state,
				/*unf=*/unf
			)
		);
		// nothing averages this sample, so an unclamped firefly stays a
		// blown pixel for the whole drag
		imageStore(
			vkofImages[pc.resolve.outputImageHandle],
			coord,
			f32v4(
				fnFireflyClamp(viewIrradiance.rgb, unf.fireflyClampLuminance),
				viewIrradiance.a
			)
		);
		return;
	}
	// (TODO REVIEW)

	UtilVdb vdb;
	vdb.isValid = false;
	if (unf.vdb != 0u) {
		// unf.vdb points at the VdbHandle descriptor (grid address +
		// densityMax), not at the raw grid blob
		vdb = utilVdbBindHandle(unf.vdb, unf.vdbWorldToLocal);
		vdb.isValid = true;
	}

	// material inspector probe: unf.probePixel is (-1,-1) on every frame
	// except the one right after a right-click, so this only ever fires
	// for the single invocation whose pixel the user actually clicked
	if (all(coord == unf.probePixel)) {
		// first ray query into physical surfaces
		RayQueryResult primaryHit = (
			utilTraceRay(
				pc.global.cameraPos,
				rayDir,
				999999.0f,
				/*isShadowRay=*/false,
				/*testAlphaCutoff=*/true
			)
		);
		// utilTraceRay's miss case returns dist=0 with modelDrawIndex=0 --
		// which happens to look like a real hit against instance 0, so a
		// click on empty space needs its own explicit sentinel rather than
		// reporting whatever garbage instance-0-indexed data would produce
		GpuProbeResult result = GpuProbeResult(-1, -1);
		if (primaryHit.dist > 0.0f) {
			UtilMeshAttributeDataFromIndices rqData = (
				utilMeshAttributeDataFromIndices(
					pc.global.models,
					primaryHit.modelDrawIndex,
					primaryHit.primitiveIndex,
					primaryHit.barycentric,
					/*unpackUvDerivatives=*/true
				)
			);
			// modelDrawIndex distinguishes which stage instance was hit
			// (file view only ever has one, so it's always 0 there);
			// materialIndex is scoped to that instance's own material table
			result = GpuProbeResult(
				i32(primaryHit.modelDrawIndex), i32(rqData.materialIndex)
			);
		}
		// a plain buffer write, not debug_printf -- the cpu reads this back
		// directly, no validation layers/probe_message plumbing needed
		((GpuProbeResult *)(unf.probeResult))[0] = result;
	}

	// -- g-buffer for pt-denoise.comp: cheap dedicated primary-ray trace,
	// not part of light transport
	{
		RayQueryResult gbufferHit = (
			utilTraceRay(
				pc.global.cameraPos,
				rayDir,
				999999.0f,
				/*isShadowRay=*/false,
				/*testAlphaCutoff=*/true
			)
		);
		f32v3 gbufferNormal = f32v3(0.0f);
		f32v4 gbufferPos = f32v4(0.0f);
		// no surface to demodulate against on a miss -- 1.0 makes the
		// denoiser's albedo divide a no-op there
		f32v3 gbufferAlbedo = f32v3(1.0f);
		if (gbufferHit.dist > 0.0f) {
			UtilMeshAttributeDataFromIndices gbufferData = (
				utilMeshAttributeDataFromIndices(
					pc.global.models,
					gbufferHit.modelDrawIndex,
					gbufferHit.primitiveIndex,
					gbufferHit.barycentric,
					/*unpackUvDerivatives=*/false
				)
			);
			// smooth normal, not normalGeometrical -- flat per-triangle
			// normals facet the denoiser's edge-stopping weight
			gbufferNormal = gbufferData.normal;
			// w = hit flag, so the denoiser can tell surface from miss
			gbufferPos = (
				f32v4(pc.global.cameraPos + rayDir * gbufferHit.dist, 1.0f)
			);
			f32v3 gbufferGeometryNormal, gbufferGeometryCoatNormal;
			f32v4 gbufferGeometryTangent, gbufferGeometryCoatTangent;
			const OpenPbrMaterial gbufferMat = (
				openPbrLoadMaterialDeriv(
					gbufferData.materialBuf,
					gbufferData.materialIndex,
					gbufferData.uvTransforms,
					gbufferData.uv,
					f32v2(0.0f),
					f32v2(0.0f),
					gbufferGeometryNormal,
					gbufferGeometryTangent,
					gbufferGeometryCoatNormal,
					gbufferGeometryCoatTangent
				)
			);
			gbufferAlbedo = gbufferMat.baseColor;
		}
		imageStore(vkofImages[pc.resolve.gbufferPosHandle], coord, gbufferPos);
		imageStore(
			vkofImages[pc.resolve.gbufferNorHandle], coord,
			f32v4(gbufferNormal, 0.0f)
		);
		imageStore(
			vkofImages[pc.resolve.gbufferAlbedoHandle], coord,
			f32v4(gbufferAlbedo, 0.0f)
		);
	}

	const f32v4 irradiance = (
		utilIrradiancePropagationEntry(
			/*tables=*/materialTables,
			/*vdb=*/vdb,
			/*origin=*/pc.global.cameraPos,
			/*wi=*/rayDir,
			/*propagationDepth=*/8u,
			/*state=*/state,
			/*unf=*/unf
		)
	);

	// (TODO REVIEW)
	const f32v3 clampedRgb = (
		fnFireflyClamp(irradiance.rgb, unf.fireflyClampLuminance)
	);
	// (TODO REVIEW)

	imageStore(
		vkofImages[pc.resolve.outputImageHandle],
		coord,
		f32v4(clampedRgb, irradiance.a)
	);
}
