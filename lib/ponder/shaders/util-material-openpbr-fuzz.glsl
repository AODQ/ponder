#ifndef UTIL_MATERIAL_OPENPBR_FUZZ_GLSL
#define UTIL_MATERIAL_OPENPBR_FUZZ_GLSL

// returns (a, b, R) from the zeltner sheen LTC parameter table;
// zeltner, burley & chiang 2022 "practical multiple-scattering sheen using
// linearly transformed cosines": a 32x32 fit over (mu, roughness) of the
// LTC matrix entries plus the directional albedo R of the fitted
// volumetric SGGX fiber layer. table data lives in app/material-tables.cpp
vec3 utilZeltnerFuzzLookup(
	const MaterialTableHandles tables,
	const float mu,
	const float roughness
) {
	// mu axis: (i+0.5f)/32.0f (grazing->normal)
	// roughness axis: (i/32.0f)^2 (smooth->rough)
	const float ai = clamp(roughness * 31.0f, 0.0f, 30.999f);
	const float mi = clamp(mu*32.0f - 0.5f, 0.0f, 30.999f);
	const int ai0 = int(ai);
	const int ai1 = min(ai0 + 1, 31);
	const int mi0 = int(mi);
	const int mi1 = min(mi0 + 1, 31);
	const float ta = fract(ai);
	const float tm = fract(mi);
	return (
		mix(
			mix(
				texelFetch(
					vkofTextures[tables.zeltnerLtcParam], ivec2(mi0, ai0), 0
				).rgb,
				texelFetch(
					vkofTextures[tables.zeltnerLtcParam], ivec2(mi1, ai0), 0
				).rgb,
				ta
			),
			mix(
				texelFetch(
					vkofTextures[tables.zeltnerLtcParam], ivec2(mi0, ai1), 0
				).rgb,
				texelFetch(
					vkofTextures[tables.zeltnerLtcParam], ivec2(mi1, ai1), 0
				).rgb,
				ta
			),
			tm
		)
	);
}

// builds the tangent frame required by the zeltner LTC parameterization:
// z along nor, x along the projection of wi onto the tangent plane
void utilFuzzFrame(
	const vec3 nor,
	const vec3 wi,
	out vec3 tanX,
	out vec3 tanY
) {
	const vec3 tRaw = wi - nor * dot(nor, wi);
	const float tLen2 = dot(tRaw, tRaw);
	if (tLen2 < 1e-12f) {
		// wi parallel to nor: azimuth is arbitrary
		utilCalculateXy(nor, tanY, tanX);
		return;
	}
	tanX = tRaw * inversesqrt(tLen2);
	tanY = cross(nor, tanX);
}

vec3 openPbrFuzzEvaluateF(
	const MaterialTableHandles tables,
	const OpenPbrMaterial mat,
	const vec3 coatedBase,
	const vec3 nor,
	const vec3 wi,
	const vec3 wo
) {
	// sheen approximated as an LTC-transformed cosine fitted to a
	// homogeneous volumetric layer with a fiber-like SGGX microflake phase
	// function. TODO fill the eq placeholders from the paper copies
	/*
		\begin{align*}
		&\textbf{(1) ltc transform law} \\
		% a cosine distribution D_o pushed through matrix M has density
		&D(\omega) = D_o(
			\tfrac{M^{-1} \omega}{||M^{-1} \omega||}
		) \frac{|M^{-1}|}{||M^{-1} \omega||^3}
			\tag{heitz et al. 2016, eq ??}\\
		&\textbf{(2) zeltner sheen parameterization} \\
		% fitted in the frame where wi has zero azimuth (utilFuzzFrame);
		% the frame choice is part of the fit, not a convention
		&M^{-1} = [[a, 0, b], [0, a, 0], [0, 0, 1]], \quad |M^{-1}| = a^2
			\tag{zeltner et al. 2022, eq ??}\\
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
		% so t_z = cos\theta_o exactly and the cosines cancel, which is why
		% no cosine factor appears in the code below
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
	const vec3 fuzzLtcParams = (
		utilZeltnerFuzzLookup(tables, dot(nor, wi), mat.fuzzRoughness)
	);
	const float a = fuzzLtcParams.x;
	const float b = fuzzLtcParams.y;
	const float eFuzz = fuzzLtcParams.z;

	// need to apply M_i^{-1} to wo in the wi-aligned frame (zeltner 2022)
	vec3 tanX, tanY;
	utilFuzzFrame(nor, wi, tanX, tanY);
	const float dotNorWo = dot(nor, wo);
	// t = M_i^{-1} \cdot \omega_o
	const vec3 t = vec3(
		a * dot(wo, tanX) + b * dotNorWo,
		a * dot(wo, tanY),
		dotNorWo
	);
	const float lenT2 = max(dot(t, t), 1e-8f);

	// f_{fuzz}
	const vec3 fFuzz = (
		mat.fuzzColor
		* eFuzz
		* a*a * IPI / (lenT2 * lenT2)
	);

	return mat.fuzzWeight * fFuzz + (1.0f - mat.fuzzWeight * eFuzz) * coatedBase;
}

// pdf of wo for the zeltner LTC fuzz lobe (zeltner et al. 2022): sampling
// pushes an exact cosine sample through M, so the pdf is exactly the LTC
// density of the eval derivation, including its t_z = cos\theta_o
/*
	pdf(\omega_o) = D(\omega_o) = \frac{t_z \, a^2}{\pi ||t||^4}
*/
float openPbrFuzzPdf(
	const vec3 nor,
	const vec3 wi,
	const vec3 wo,
	const float fuzzA,
	const float fuzzB
) {
	// t = M_i^{-1} \cdot \omega_o in the wi-aligned frame
	vec3 tanX, tanY;
	utilFuzzFrame(nor, wi, tanX, tanY);
	const float dotNorWo = dot(nor, wo);
	const vec3 t = vec3(
		fuzzA * dot(wo, tanX) + fuzzB * dotNorWo,
		fuzzA * dot(wo, tanY),
		dotNorWo
	);
	const float lenT2 = max(dot(t, t), 1e-8f);
	return dotNorWo * fuzzA * fuzzA * IPI / (lenT2 * lenT2);
}

// samples wo from the zeltner LTC fuzz lobe; must stay the exact inverse of
// the M_i^{-1} transform used by openPbrFuzzPdf and openPbrFuzzEvaluateF
vec3 openPbrFuzzSampleWo(
	const vec3 nor,
	const vec3 wi,
	const float fuzzA,
	const float fuzzB,
	const vec2 xi
) {
	const vec3 dLocal = utilToCartesian(sqrt(xi.x), TAU * xi.y);
	// M = inv(M^{-1}) = [[1/a, 0, -b/a], [0, 1/a, 0], [0, 0, 1]]
	const float safeA = max(fuzzA, 1e-5f);
	const vec3 tLocal = vec3(
		(dLocal.x - fuzzB * dLocal.z) / safeA,
		dLocal.y / safeA,
		dLocal.z
	);
	// same wi-aligned frame as the pdf and eval (zeltner 2022)
	vec3 tanX, tanY;
	utilFuzzFrame(nor, wi, tanX, tanY);
	return (
		normalize(tanX * tLocal.x + tanY * tLocal.y + nor * tLocal.z)
	);
}

#endif // UTIL_MATERIAL_OPENPBR_FUZZ_GLSL
