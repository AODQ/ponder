implementing bsdf;

import types;
import util;
import material;

// directions in this file point away from the surface: ωi here is the
// path's -ωi (see units.slang); Bsdf_OpenPbr negates it once at entry

namespace vkof {

// returns (a, b, R) from zeltner sheen LTC parameter table
// zeltner, burley & chiang 2022
// 32x32 (mu, roughness) LTC matrix entries plus directional albedo R
public func bsdfZeltnerFuzzLookup(
	zeltnerLtcParamHandle: u32,
	μ: f32,
	roughness: f32,
) -> f32v3 {
	// mu axis: i/31 (grazing->normal)
	// roughness axis: i/31 (smooth->rough)
	let ai: f32 = clamp(roughness * 31.0f, 0.0f, 30.999f);
	let mi: f32 = clamp(μ * 31.0f, 0.0f, 30.999f);
	let ai0: i32 = i32(ai);
	let ai1: i32 = min(ai0 + 1, 31);
	let mi0: i32 = i32(mi);
	let mi1: i32 = min(mi0 + 1, 31);
	let ta: f32 = frac(ai);
	let tm: f32 = frac(mi);
	return (
		lerp(
			lerp(
				vkofTextures[zeltnerLtcParamHandle].Load(i32v3(mi0, ai0, 0)).rgb,
				vkofTextures[zeltnerLtcParamHandle].Load(i32v3(mi1, ai0, 0)).rgb,
				tm
			),
			lerp(
				vkofTextures[zeltnerLtcParamHandle].Load(i32v3(mi0, ai1, 0)).rgb,
				vkofTextures[zeltnerLtcParamHandle].Load(i32v3(mi1, ai1, 0)).rgb,
				tm
			),
			ta
		)
	);
}

// builds the tangent frame required by the zeltner LTC parameterization:
// tangent frame required by zeltner LTC parameterization
// z along normal, x along projection of ωi onto tangent plane
public func bsdfFuzzFrame(
	nor: f32v3,
	ωi: f32v3,
	out tanX: f32v3,
	out tanY: f32v3,
) {
	let tRaw: f32v3 = ωi - nor * dot(nor, ωi);
	let tLen2: f32 = dot(tRaw, tRaw);
	if (tLen2 < 1e-12f) {
		// wi parallel to nor: azimuth is arbitrary
		vkof::calculateXy(nor, tanY, tanX);
		return;
	}
	tanX = tRaw * rsqrt(tLen2);
	tanY = cross(nor, tanX);
}

public func bsdfOpenPbrFuzzEvaluateF(
	zeltnerLtcParamHandle: u32,
	mat: MaterialBackend_OpenPbr.Params,
	coatedBase: vkof::Spectrum,
	nor: f32v3,
	ωi: f32v3,
	ωo: f32v3,
	λ: Wavelengths,
) -> vkof::Spectrum {
	if (mat.fuzzWeight <= 0.0f) {
		return coatedBase;
	}

	// sheen approximated as an LTC-transformed cosine fitted to a
	// homogeneous volumetric layer with a fiber-like SGGX microflake phase
	// function.
	/*
		\begin{align*}
		&\textbf{(1) ltc transform law} \\
		% a cosine distribution D_o pushed through matrix M has density
		&D(\omega) = D_o(
			\tfrac{M^{-1} \omega}{||M^{-1} \omega||}
		) \frac{|M^{-1}|}{||M^{-1} \omega||^3}
			\tag{heitz et al. 2016, eq 1}\\
		&\textbf{(2) zeltner sheen parameterization} \\
		% fitted in the frame where wi has zero azimuth (utilFuzzFrame);
		&M^{-1} = [[a, 0, b], [0, a, 0], [0, 0, 1]], \quad |M^{-1}| = a^2
			\tag{zeltner et al. 2022, eq 1}\\
		&t = M^{-1} \omega_o = (
			a \omega_{o,x} + b \omega_{o,z}, \;
			a \omega_{o,y}, \;
			\omega_{o,z}
		)\\
		% D_o is cos/\pi of the back-transformed direction, t_z / ||t||,
		% merging with the 1/||t||^3 jacobian:
		&D(\omega_o) = \frac{t_z \, a^2}{\pi ||t||^4}\\
		&\textbf{(3) the evaluated lobe} \\
		% f cos\theta_o = C_{fuzz} R D; the matrix third row is (0, 0, 1)
		% so t_z = cos\theta_o and the cosines cancel, so not used in calculation
		&f(\omega_i, \omega_o) = C_{fuzz} R \frac{a^2}{\pi ||t||^4}\\
		&\textbf{(4) layering} \\
		% R (LUT .z) is the fitted directional albedo of the sheen lobe,
		% used both here and as the lobe selection probability
		&f_{surface} = F f_{fuzz} + (1 - F R) f_{below}
			\tag{openpbr spec, fuzz}\\
		\end{align*}
	*/

	// note that fuzz is isotropic
	// (a, b, R) at the incident cosine
	let fuzzLtcParams: f32v3 = (
		bsdfZeltnerFuzzLookup(zeltnerLtcParamHandle, dot(nor, ωi), mat.fuzzRoughness)
	);
	let a: f32 = fuzzLtcParams.x;
	let b: f32 = fuzzLtcParams.y;
	let eFuzz: f32 = fuzzLtcParams.z;

	// need to apply M_i^{-1} to ωo in the ωi-aligned frame (zeltner 2022)
	var tanX: f32v3;
	var tanY: f32v3;
	bsdfFuzzFrame(nor, ωi, tanX, tanY);
	let dotNorWo: f32 = dot(nor, ωo);
	// t = M_i^{-1} \cdot \omega_o
	let t: f32v3 = (
		f32v3(
			a * dot(ωo, tanX) + b * dotNorWo,
			a * dot(ωo, tanY),
			dotNorWo
		)
	);
	let lenT2: f32 = max(dot(t, t), 1e-8f);

	// f_{fuzz}
	let fFuzz: vkof::Spectrum = (
		mat.fuzzColor
		* (eFuzz * a*a * (1.0f / π) / (lenT2 * lenT2))
	);

	return (
		mat.fuzzWeight * fFuzz + (1.0f - mat.fuzzWeight * eFuzz) * coatedBase
	);
}

// pdf of ωo for the zeltner LTC fuzz lobe (zeltner et al. 2022)
/*
	pdf(\omega_o) = D(\omega_o) = \frac{t_z \, a^2}{\pi ||t||^4}
*/
public func bsdfOpenPbrFuzzPdf(
	nor: f32v3,
	ωi: f32v3,
	ωo: f32v3,
	fuzzA: f32,
	fuzzB: f32,
) -> f32 {
	// t = M_i^{-1} \cdot \omega_o in the wi-aligned frame
	var tanX: f32v3;
	var tanY: f32v3;
	bsdfFuzzFrame(nor, ωi, tanX, tanY);
	let dotNorWo: f32 = dot(nor, ωo);
	let t: f32v3 = f32v3(
		fuzzA * dot(ωo, tanX) + fuzzB * dotNorWo,
		fuzzA * dot(ωo, tanY),
		dotNorWo
	);
	let lenT2: f32 = max(dot(t, t), 1e-8f);
	return max(dotNorWo, 0.0f) * fuzzA * fuzzA * (1.0f / π) / (lenT2 * lenT2);
}

// samples ωo from the zeltner LTC fuzz lobe
public func bsdfOpenPbrFuzzSampleWo(
	nor: f32v3,
	ωi: f32v3,
	fuzzA: f32,
	fuzzB: f32,
	ξ: f32v2,
) -> f32v3 {
	let dLocal: f32v3 = vkof::toCartesian(sqrt(ξ.x), 2.0f * π * ξ.y);
	// M = inv(M^{-1}) = [[1/a, 0, -b/a], [0, 1/a, 0], [0, 0, 1]]
	let safeA: f32 = max(fuzzA, 1e-5f);
	let tLocal: f32v3 = f32v3(
		(dLocal.x - fuzzB * dLocal.z) / safeA,
		dLocal.y / safeA,
		dLocal.z
	);
	// same wi-aligned frame as the pdf and eval (zeltner 2022)
	var tanX: f32v3;
	var tanY: f32v3;
	bsdfFuzzFrame(nor, ωi, tanX, tanY);
	return (
		normalize(tanX * tLocal.x + tanY * tLocal.y + nor * tLocal.z)
	);
}

} // namespace vkof
