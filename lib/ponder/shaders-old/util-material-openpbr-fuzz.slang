module "util-material-openpbr-fuzz";
__exported import "util-types";
__exported import "util-material-openpbr-shared";
__exported import "util-material-openpbr-tables";
__exported import "util-shading-frame";

// returns (a, b, R) from the zeltner sheen LTC parameter table;
// zeltner, burley & chiang 2022
// a 32x32 fit over (mu, roughness) of the LTC matrix entries plus the
// directional albedo R of the fitted volumetric SGGX fiber layer.
public func utilZeltnerFuzzLookup(
	tables: MaterialTableHandles,
	mu: f32,
	roughness: f32,
) -> f32v3 {
	// mu axis: (i+0.5f)/32.0f (grazing->normal)
	// roughness axis: (i/32.0f)^2 (smooth->rough)
	const f32 ai = clamp(roughness * 31.0f, 0.0f, 30.999f);
	const f32 mi = clamp(mu*32.0f - 0.5f, 0.0f, 30.999f);
	const i32 ai0 = i32(ai);
	const i32 ai1 = min(ai0 + 1, 31);
	const i32 mi0 = i32(mi);
	const i32 mi1 = min(mi0 + 1, 31);
	const f32 ta = frac(ai);
	const f32 tm = frac(mi);
	return (
		lerp(
			lerp(
				texelFetch(
					vkofTextures[tables.zeltnerLtcParam], i32v2(mi0, ai0), 0
				).rgb,
				texelFetch(
					vkofTextures[tables.zeltnerLtcParam], i32v2(mi1, ai0), 0
				).rgb,
				ta
			),
			lerp(
				texelFetch(
					vkofTextures[tables.zeltnerLtcParam], i32v2(mi0, ai1), 0
				).rgb,
				texelFetch(
					vkofTextures[tables.zeltnerLtcParam], i32v2(mi1, ai1), 0
				).rgb,
				ta
			),
			tm
		)
	);
}

// builds the tangent frame required by the zeltner LTC parameterization:
// z along nor, x along the projection of wi onto the tangent plane
public func utilFuzzFrame(
	nor: f32v3,
	wi: f32v3,
	out tanX: f32v3,
	out tanY: f32v3,
) {
	const f32v3 tRaw = wi - nor * dot(nor, wi);
	const f32 tLen2 = dot(tRaw, tRaw);
	if (tLen2 < 1e-12f) {
		// wi parallel to nor: azimuth is arbitrary
		utilCalculateXy(nor, tanY, tanX);
		return;
	}
	tanX = tRaw * rsqrt(tLen2);
	tanY = cross(nor, tanX);
}

public func openPbrFuzzEvaluateF(
	tables: MaterialTableHandles,
	mat: OpenPbrMaterial,
	coatedBase: f32v3,
	nor: f32v3,
	wi: f32v3,
	wo: f32v3,
) -> f32v3 {
	// return value reduces to exactly coatedBase at fuzzWeight == 0 (see the
	// mix at the bottom); skip the LUT fetch + LTC transform for it
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
	const f32v3 fuzzLtcParams = (
		utilZeltnerFuzzLookup(tables, dot(nor, wi), mat.fuzzRoughness)
	);
	const f32 a = fuzzLtcParams.x;
	const f32 b = fuzzLtcParams.y;
	const f32 eFuzz = fuzzLtcParams.z;

	// need to apply M_i^{-1} to wo in the wi-aligned frame (zeltner 2022)
	f32v3 tanX, tanY;
	utilFuzzFrame(nor, wi, tanX, tanY);
	const f32 dotNorWo = dot(nor, wo);
	// t = M_i^{-1} \cdot \omega_o
	const f32v3 t = (
		f32v3(
			a * dot(wo, tanX) + b * dotNorWo,
			a * dot(wo, tanY),
			dotNorWo
		)
	);
	const f32 lenT2 = max(dot(t, t), 1e-8f);

	// f_{fuzz}
	const f32v3 fFuzz = (
		mat.fuzzColor
		* eFuzz
		* a*a * IPI / (lenT2 * lenT2)
	);

	return mat.fuzzWeight * fFuzz + (1.0f - mat.fuzzWeight * eFuzz) * coatedBase;
}

// pdf of wo for the zeltner LTC fuzz lobe (zeltner et al. 2022)
/*
	pdf(\omega_o) = D(\omega_o) = \frac{t_z \, a^2}{\pi ||t||^4}
*/
public func openPbrFuzzPdf(
	nor: f32v3,
	wi: f32v3,
	wo: f32v3,
	fuzzA: f32,
	fuzzB: f32,
) -> f32 {
	// t = M_i^{-1} \cdot \omega_o in the wi-aligned frame
	f32v3 tanX, tanY;
	utilFuzzFrame(nor, wi, tanX, tanY);
	const f32 dotNorWo = dot(nor, wo);
	const f32v3 t = f32v3(
		fuzzA * dot(wo, tanX) + fuzzB * dotNorWo,
		fuzzA * dot(wo, tanY),
		dotNorWo
	);
	const f32 lenT2 = max(dot(t, t), 1e-8f);
	return max(dotNorWo, 0.0f) * fuzzA * fuzzA * IPI / (lenT2 * lenT2);
}

// samples wo from the zeltner LTC fuzz lobe; must stay the exact inverse of
// the M_i^{-1} transform used by openPbrFuzzPdf and openPbrFuzzEvaluateF
public func openPbrFuzzSampleWo(
	nor: f32v3,
	wi: f32v3,
	fuzzA: f32,
	fuzzB: f32,
	xi: f32v2,
) -> f32v3 {
	const f32v3 dLocal = utilToCartesian(sqrt(xi.x), TAU * xi.y);
	// M = inv(M^{-1}) = [[1/a, 0, -b/a], [0, 1/a, 0], [0, 0, 1]]
	const f32 safeA = max(fuzzA, 1e-5f);
	const f32v3 tLocal = f32v3(
		(dLocal.x - fuzzB * dLocal.z) / safeA,
		dLocal.y / safeA,
		dLocal.z
	);
	// same wi-aligned frame as the pdf and eval (zeltner 2022)
	f32v3 tanX, tanY;
	utilFuzzFrame(nor, wi, tanX, tanY);
	return (
		normalize(tanX * tLocal.x + tanY * tLocal.y + nor * tLocal.z)
	);
}

