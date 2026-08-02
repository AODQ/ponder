import "util-shading-frame";
import "util-material-openpbr";
import "util-material-openpbr-load";
import "util-mor-shared";

struct PC {
	u32v4 rootPad[8];
	u64 inVa;
	u64 outVa;
	u32 threadCount;
	u32 pad;
	u64 materialsVa;
	u64 uvTransformsVa;
};

[[vk::push_constant]] PC pc;

// -load.glsl reads skMinRoughness and OpenPbrMaterial from its includer,
// exactly as util-raytrace.glsl supplies them

// the X-macro expansion in util-material-openpbr-load, against a real mor
// material table. every parameter goes through LOAD_COMPONENT1/3/NORMAL, so
// a mis-expansion on either side shows up in whichever field it touched

// in:  materialIndex, uv(2), uvDx(2), uvDy(2), lod
// out: 21 loaded parameters, then the four geometry outputs
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
	const u32 o = thread * 35u;

	GpuMorMaterial * materials = (GpuMorMaterial *)(pc.materialsVa);
	GpuMorUvTransform * uvTransforms = (
		(GpuMorUvTransform *)(pc.uvTransformsVa)
	);
	const u32 idx = u32(inBuf[i + 0u]);
	const f32v2 uv = f32v2(inBuf[i + 1u], inBuf[i + 2u]);
	const f32v2 uvDx = f32v2(inBuf[i + 3u], inBuf[i + 4u]);
	const f32v2 uvDy = f32v2(inBuf[i + 5u], inBuf[i + 6u]);

	f32v3 geometryNormal;
	f32v4 geometryTangent;
	f32v3 geometryCoatNormal;
	f32v4 geometryCoatTangent;
	const OpenPbrMaterial mat = openPbrLoadMaterialDeriv(
		materials, idx, uvTransforms, uv, uvDx, uvDy,
		geometryNormal, geometryTangent, geometryCoatNormal, geometryCoatTangent
	);

	outBuf[o + 0u] = mat.baseWeight;
	outBuf[o + 1u] = mat.baseColor.r;
	outBuf[o + 2u] = mat.baseColor.g;
	outBuf[o + 3u] = mat.baseColor.b;
	outBuf[o + 4u] = mat.baseMetalness;
	outBuf[o + 5u] = mat.baseDiffuseRoughness;
	outBuf[o + 6u] = mat.specularWeight;
	outBuf[o + 7u] = mat.specularRoughness;
	outBuf[o + 8u] = mat.specularRoughnessAnisotropy;
	outBuf[o + 9u] = mat.specularIor;
	outBuf[o + 10u] = mat.transmissionWeight;
	outBuf[o + 11u] = mat.transmissionDepth;
	outBuf[o + 12u] = mat.transmissionScatterAnisotropy;
	outBuf[o + 13u] = mat.subsurfaceWeight;
	outBuf[o + 14u] = mat.subsurfaceRadius;
	outBuf[o + 15u] = mat.coatWeight;
	outBuf[o + 16u] = mat.coatRoughness;
	outBuf[o + 17u] = mat.coatIor;
	outBuf[o + 18u] = mat.fuzzWeight;
	outBuf[o + 19u] = mat.fuzzRoughness;
	outBuf[o + 20u] = mat.emissionLuminance;
	outBuf[o + 21u] = mat.emissionColor.r;
	outBuf[o + 22u] = mat.thinFilmWeight;
	outBuf[o + 23u] = mat.thinFilmThickness;
	outBuf[o + 24u] = mat.thinFilmIor;
	outBuf[o + 25u] = mat.geometryOpacity;

	outBuf[o + 26u] = geometryNormal.x;
	outBuf[o + 27u] = geometryNormal.y;
	outBuf[o + 28u] = geometryNormal.z;
	outBuf[o + 29u] = geometryTangent.x;
	outBuf[o + 30u] = geometryTangent.w;
	outBuf[o + 31u] = geometryCoatNormal.x;
	outBuf[o + 32u] = geometryCoatNormal.z;
	outBuf[o + 33u] = geometryCoatTangent.y;
	outBuf[o + 34u] = geometryCoatTangent.w;
}
