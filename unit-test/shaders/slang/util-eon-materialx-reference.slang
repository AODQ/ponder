#pragma once
__exported import "util-types";


// independent cross-check only, not used by the real ponder implementation.
// ported from AcademySoftwareFoundation/MaterialX
// libraries/pbrlib/genglsl/lib/mx_microfacet_diffuse.glsl

// mx_math.glsl support, reduced to what the diffuse functions below use
public static const f32 mxPi = 3.1415926535897932;
public static const f32 mxPiInv = 1.0f / mxPi;
public static const f32 mxFloatEps = 1e-8f;
public func mxSquare(
	x: f32,
) -> f32 {
	// glsl parameters are mutable copies; slang's are not
	var x = x; return x * x; }
public func mxSquare(
	x: f32v3,
) -> f32v3 {
	// glsl parameters are mutable copies; slang's are not
	var x = x; return x * x; }
public func mxAcos(
	x: f32,
) -> f32 {
	// glsl parameters are mutable copies; slang's are not
	var x = x; return acos(x); }

public static const f32 mxFujiiConstant1 = 0.5f - 2.0f / (3.0f * mxPi);
public static const f32 mxFujiiConstant2 = 2.0f / 3.0f - 28.0f / (15.0f * mxPi);

// Improved Oren-Nayar diffuse from Fujii:
// https://mimosa-pudica.net/improved-oren-nayar.html
// note: exact closed-form E_FON, unlike ponder's portsmouth polynomial fit.
// undefined (0/0) at cosTheta == 0 exactly; callers stay off exact grazing
public func mxOrenNayarFujiiDiffuseDirAlbedo(
	cosTheta: f32,
	roughness: f32,
) -> f32 {
	f32 A = 1.0f / (1.0f + mxFujiiConstant1 * roughness);
	f32 B = roughness * A;
	f32 Si = sqrt(max(0.0f, 1.0f - mxSquare(cosTheta)));
	f32 G = Si * (mxAcos(clamp(cosTheta, -1.0f, 1.0f)) - Si * cosTheta) +
	        2.0f * ((Si / cosTheta) * (1.0f - Si * Si * Si) - Si) / 3.0f;
	return A + (B * G * mxPiInv);
}

public func mxOrenNayarFujiiDiffuseAvgAlbedo(
	roughness: f32,
) -> f32 {
	// glsl parameters are mutable copies; slang's are not
	var roughness = roughness;
	f32 A = 1.0f / (1.0f + mxFujiiConstant1 * roughness);
	return A * (1.0f + mxFujiiConstant2 * roughness);
}

// Energy-compensated Oren-Nayar diffuse from OpenPBR Surface:
// https://academysoftwarefoundation.github.io/OpenPBR/
// returns the brdf without the 1/pi normalization and without any diffuse
// weight; ponder's openPbrGlossyDiffuseOrenNayerEvaluateF equals
// baseWeight * (1/pi) * this, with color = baseColor
public func mxOrenNayarCompensatedDiffuse(
	NdotV: f32,
	NdotL: f32,
	LdotV: f32,
	roughness: f32,
	color: f32v3,
) -> f32v3 {
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

public func mxOrenNayarCompensatedDiffuseDirAlbedo(
	cosTheta: f32,
	roughness: f32,
	color: f32v3,
) -> f32v3 {
	f32 dirAlbedo = mxOrenNayarFujiiDiffuseDirAlbedo(cosTheta, roughness);
	f32 avgAlbedo = mxOrenNayarFujiiDiffuseAvgAlbedo(roughness);
	f32v3 colorMultiScatter = mxSquare(color) * avgAlbedo /
	                          (f32v3(1.0f) - color * max(0.0f, 1.0f - avgAlbedo));
	return lerp(colorMultiScatter, color, dirAlbedo);
}

