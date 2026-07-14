#ifndef UTIL_EON_MATERIALX_REFERENCE_GLSL
#define UTIL_EON_MATERIALX_REFERENCE_GLSL

#ifndef f32
#define i32v2 ivec2
#define f32 float
#define f32v2 vec2
#define f32v3 vec3
#define u32 uint
#endif

// independent cross-check only, not used by the real ponder implementation.
// ported from AcademySoftwareFoundation/MaterialX
// libraries/pbrlib/genglsl/lib/mx_microfacet_diffuse.glsl

// mx_math.glsl support, reduced to what the diffuse functions below use
const f32 mxPi = 3.1415926535897932;
const f32 mxPiInv = 1.0f / mxPi;
const f32 mxFloatEps = 1e-8f;
f32 mxSquare(f32 x) { return x * x; }
f32v3 mxSquare(f32v3 x) { return x * x; }
f32 mxAcos(f32 x) { return acos(x); }

const f32 mxFujiiConstant1 = 0.5f - 2.0f / (3.0f * mxPi);
const f32 mxFujiiConstant2 = 2.0f / 3.0f - 28.0f / (15.0f * mxPi);

// Improved Oren-Nayar diffuse from Fujii:
// https://mimosa-pudica.net/improved-oren-nayar.html
// note: exact closed-form E_FON, unlike ponder's portsmouth polynomial fit.
// undefined (0/0) at cosTheta == 0 exactly; callers stay off exact grazing
f32 mxOrenNayarFujiiDiffuseDirAlbedo(f32 cosTheta, f32 roughness) {
	f32 A = 1.0f / (1.0f + mxFujiiConstant1 * roughness);
	f32 B = roughness * A;
	f32 Si = sqrt(max(0.0f, 1.0f - mxSquare(cosTheta)));
	f32 G = Si * (mxAcos(clamp(cosTheta, -1.0f, 1.0f)) - Si * cosTheta) +
	        2.0f * ((Si / cosTheta) * (1.0f - Si * Si * Si) - Si) / 3.0f;
	return A + (B * G * mxPiInv);
}

f32 mxOrenNayarFujiiDiffuseAvgAlbedo(f32 roughness) {
	f32 A = 1.0f / (1.0f + mxFujiiConstant1 * roughness);
	return A * (1.0f + mxFujiiConstant2 * roughness);
}

// Energy-compensated Oren-Nayar diffuse from OpenPBR Surface:
// https://academysoftwarefoundation.github.io/OpenPBR/
// returns the brdf without the 1/pi normalization and without any diffuse
// weight; ponder's openPbrGlossyDiffuseOrenNayerEvaluateF equals
// baseWeight * (1/pi) * this, with color = baseColor
f32v3 mxOrenNayarCompensatedDiffuse(
	f32 NdotV, f32 NdotL, f32 LdotV, f32 roughness, f32v3 color
) {
	f32 s = LdotV - NdotL * NdotV;
	f32 stinv = (s > 0.0f) ? s / max(NdotL, NdotV) : s;

	// Compute the single-scatter lobe.
	f32 A = 1.0f / (1.0f + mxFujiiConstant1 * roughness);
	f32v3 lobeSingleScatter = color * A * (1.0f + roughness * stinv);

	// Compute the multi-scatter lobe.
	f32 dirAlbedoV = mxOrenNayarFujiiDiffuseDirAlbedo(NdotV, roughness);
	f32 dirAlbedoL = mxOrenNayarFujiiDiffuseDirAlbedo(NdotL, roughness);
	f32 avgAlbedo = mxOrenNayarFujiiDiffuseAvgAlbedo(roughness);
	f32v3 colorMultiScatter = mxSquare(color) * avgAlbedo /
	                          (f32v3(1.0f) - color * max(0.0f, 1.0f - avgAlbedo));
	f32v3 lobeMultiScatter = colorMultiScatter *
	                         max(mxFloatEps, 1.0f - dirAlbedoV) *
	                         max(mxFloatEps, 1.0f - dirAlbedoL) /
	                         max(mxFloatEps, 1.0f - avgAlbedo);

	// Return the sum.
	return lobeSingleScatter + lobeMultiScatter;
}

f32v3 mxOrenNayarCompensatedDiffuseDirAlbedo(
	f32 cosTheta, f32 roughness, f32v3 color
) {
	f32 dirAlbedo = mxOrenNayarFujiiDiffuseDirAlbedo(cosTheta, roughness);
	f32 avgAlbedo = mxOrenNayarFujiiDiffuseAvgAlbedo(roughness);
	f32v3 colorMultiScatter = mxSquare(color) * avgAlbedo /
	                          (f32v3(1.0f) - color * max(0.0f, 1.0f - avgAlbedo));
	return mix(colorMultiScatter, color, dirAlbedo);
}

#endif
