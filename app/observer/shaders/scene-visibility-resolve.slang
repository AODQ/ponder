import "util-mesh";
import "util-environment-map";
import "util-material-openpbr";
import "util-material-openpbr-load";
import "util-material-openpbr-ibl";
import "util-tonemap";
import "util-mor-shared";
import "util-scene-visibility-resolve-pc";
import "util-observer-global-pc";

struct PC {
	GpuGlobalPc global;
	GpuResolvePc resolve;
};

[[vk::push_constant]] PC pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const i32v2 coord = i32v2(gl_GlobalInvocationID.xy);
	const i32v2 screensize = (
		i32v2(pc.global.renderWidth, pc.global.renderHeight)
	);

	if (coord.x >= screensize.x || coord.y >= screensize.y) {
		return;
	}

	const u32 vis = (
		imageLoad(vkofImagesU[pc.resolve.visibilityImageHandle], coord).r
	);
	if (vis == 0u || vis == 0xFFFFFFFFu) {
		f32v3 background = f32v3(0.0f);
		if (pc.resolve.envRadianceHandle != 0u) {
			const f32v2 ndcXy = (
				(f32v2(coord) + 0.5f) / f32v2(screensize) * 2.0f - 1.0f
			);
			const f32m4x4 invViewProj = inverse(pc.global.viewProj);
			const f32v4 p0 = mul(invViewProj, f32v4(ndcXy, 0.0f, 1.0f));
			const f32v4 p1 = mul(invViewProj, f32v4(ndcXy, 1.0f, 1.0f));
			const f32v3 rayDir = normalize(p1.xyz / p1.w - p0.xyz / p0.w);
			const EnvironmentMapHandles envHandles = (
				EnvironmentMapHandles(pc.resolve.envRadianceHandle, 0u, 0u, 0u)
			);
			background = environmentMapRadiance(envHandles, rayDir, 0.0f);
		}
		imageStore(
			vkofImages[pc.resolve.outputImageHandle],
			coord,
			f32v4(fnAces(background * pc.global.exposure), 1.0f)
		);
		if (all(coord == pc.resolve.probePixel)) {
			((GpuProbeResult *)(pc.resolve.probeResultVa))[0] = (
				GpuProbeResult(-1, -1)
			);
		}
		return;
	}

	const UtilMeshUnpacked texel = utilMeshUnpackTexel(vis);
	UtilMeshAttributeData attrData = (
		utilMeshDecodeAttributes(
			UtilMeshDecodeContext(
				pc.global.models, pc.global.viewProj,
				gl_GlobalInvocationID.xy
			),
			texel, screensize,
			/*unpackWorldOrigin=*/ true,
			/*unpackNormal=*/ true,
			/*unpackUv=*/ true,
			/*unpackUvDerivatives=*/ true,
			/*unpackMaterial=*/ true
		)
	);

	if (all(coord == pc.resolve.probePixel)) {
		((GpuProbeResult *)(pc.resolve.probeResultVa))[0] = (
			GpuProbeResult(i32(texel.modelId), i32(attrData.materialIndex))
		);
	}

	const f32m3x3 tbn = (
		utilCalculateTbnBasis(attrData.normal, attrData.tangent)
	);

	f32v3 geometryNormal = f32v3(0.0f);
	f32v4 geometryTangent = f32v4(0.0f);
	f32v3 geometryCoatNormal = f32v3(0.0f);
	f32v4 geometryCoatTangent = f32v4(0.0f);
	const OpenPbrMaterial material = (
		openPbrLoadMaterialDeriv(
			attrData.materialBuf,
			attrData.materialIndex,
			attrData.uvTransforms,
			attrData.uv,
			attrData.uvDx,
			attrData.uvDy,
			geometryNormal,
			geometryTangent,
			geometryCoatNormal,
			geometryCoatTangent
		)
	);

	// (TODO REVIEW)
	// openpbr geometry_tangent/geometry_coat_tangent steer the anisotropy
	// direction; unbound decodes to +x, i.e. back to the mesh/uv tangent
	const ShadingFrame frame = (
		shadingFrameFromTangent(
			utilShadingNormalOrGeometric(
				mul(tbn, geometryNormal), attrData.normalGeometrical
			),
			mul(tbn, geometryTangent.xyz))
	);
	const ShadingFrame coatFrame = (
		shadingFrameFromTangent(
			utilShadingNormalOrGeometric(
				mul(tbn, geometryCoatNormal), attrData.normalGeometrical
			),
			mul(tbn, geometryCoatTangent.xyz))
	);
	// (TODO REVIEW)

	const f32v3 wi = normalize(pc.global.cameraPos - attrData.origin);
	const MaterialTableHandles tables = (
		MaterialTableHandles(
			pc.resolve.kullaContyEnergyHandle, pc.resolve.zeltnerLtcParamHandle
		)
	);
	// lit purely by the environment; the env map's own sun is already in the
	// prefiltered ibl, so there is no analytic directional term to add
	f32v3 color = f32v3(0.0f);

	if (pc.resolve.envIblIrradianceHandle != 0u) {
		const f32 dotNorWi = max(dot(frame.nor, wi), 0.0f);
		const f32 dotCoatNorWi = max(dot(coatFrame.nor, wi), 0.0f);
		const f32 roughenedSpecularRoughness = (
			openPbrCoatRoughenedRoughness(
				material.specularRoughness, material.coatRoughness,
				material.coatWeight
			)
		);

		const f32v3 irradiance = (
			textureLod(
				vkofTextures[pc.resolve.envIblIrradianceHandle],
				environmentMapDirToUv(frame.nor, 0.0f),
				0.0f
			).rgb
		);
		const f32v3 reflectBase = reflect(-wi, frame.nor);
		const f32v3 prefilteredSpecularBase = (
			textureLod(
				vkofTextures[pc.resolve.envIblSpecularHandle],
				environmentMapDirToUv(reflectBase, 0.0f),
				roughenedSpecularRoughness * pc.resolve.envIblSpecularMaxLod
			).rgb
		);
		const f32v3 reflectCoat = reflect(-wi, coatFrame.nor);
		const f32v3 prefilteredSpecularCoat = (
			textureLod(
				vkofTextures[pc.resolve.envIblSpecularHandle],
				environmentMapDirToUv(reflectCoat, 0.0f),
				material.coatRoughness * pc.resolve.envIblSpecularMaxLod
			).rgb
		);

		color += (
			openPbrIblEvaluateF(
				tables,
				material,
				dotNorWi,
				dotCoatNorWi,
				irradiance,
				prefilteredSpecularBase,
				prefilteredSpecularCoat,
				/*isInsideMedium=*/ false
			)
		);
	}

	imageStore(
		vkofImages[pc.resolve.outputImageHandle],
		coord,
		f32v4(fnAces(color * pc.global.exposure), 1.0f)
	);
}
