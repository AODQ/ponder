#ifndef UTIL_ENERGY_COMPENSATION_GLSL
#define UTIL_ENERGY_COMPENSATION_GLSL

#ifndef f32
#define i32v2 ivec2
#define i32 int
#define f32 float
#define f32v2 vec2
#define f32v3 vec3
#define f32v4 vec4
#define f32m44 mat4
#define u32 uint
#define u32v2 uvec2
#define u32v3 uvec3
#define u32v4 uvec4
#define u64 uint64_t
#endif

// rational quadratic fit to monte carla data for ggx directional albedo;
// from materialx
f32 utilMicrofacetGgxDirectionalAlbedo(
	const f32 dotNorWi,
	const f32 alpha,
	const f32 f0,
	const f32 f90
) {
	const f32 x = dotNorWi;
	const f32 y = alpha;
	const f32 x2 = x * x;
	const f32 y2 = y * y;
	const f32v4 r = (
		f32v4(0.1003, 0.9345, 1.0, 1.0)
		+ f32v4(-0.6303, -2.323, -1.765, 0.2281) * x
		+ f32v4(9.748, 2.229, 8.263, 15.94) * y
		+ f32v4(-2.038, -3.748, 11.53, -55.83) * x * y
		+ f32v4(29.34, 1.424, 28.96, 13.08) * x2
		+ f32v4(-8.245, -0.7684, -7.507, 41.26) * y2
		+ f32v4(-26.44, 1.436, -36.11, 54.9) * x2 * y
		+ f32v4(19.99, 0.2913, 15.86, 300.2) * x * y2
		+ f32v4(-5.448, 0.6286, 33.37, -285.1) * x2 * y2
	);
	const f32v2 ab = clamp(r.xy/r.zw, 0.0f, 1.0f);
	return f0 * ab.x + f90 * ab.y;
}

// turquin 2019 multiplicative energy compensation for the single-scatter
// ggx dielectric lobe (kulla & conty 2017 compensate additively with
// tables instead; see utilMicrofacetDielectricEnergyCompensationKullaConty)
/*
	\begin{align*}
	&\textbf{(1) compensation factor} \\
	&f = f_{ss} (1 + F_{avg} \frac{1 - E_{ss}(\mu)}{E_{ss}(\mu)})
		\tag{turquin 2019}\\
	&\textbf{(2) average fresnel} \\
	&F_{avg} = 2 \int_0^1 F(\mu) \mu \, d\mu
		= F_0 + \frac{F_{90} - F_0}{21}
		\tag{kulla \& conty 2017}\\
	\end{align*}
*/
f32 utilMicrofacetDielectricEnergyCompensate(
	const f32 mu,
	const f32 roughness,
	const f32 f0
) {
	// (2) F_{avg} = F_0 + (F_{90} - F_0) / 21, with F_{90} = 1
	const f32 fAvg = f0 + (1.0f - f0) * (1.0f / 21.0f);
	// (3) E_{ss}(\mu, \alpha), perfect fresnel; floored away from 0 -- see
	// cull's own note: the materialx rational fit legitimately lands on
	// exactly 0 at grazing mu / high roughness corners outside its
	// trained domain, and (1 - ess) / ess would otherwise be a live
	// division by zero that poisons the whole lobe with nan
	const f32 ess = (
		max(utilMicrofacetGgxDirectionalAlbedo(mu, roughness, 1.0f, 1.0f), 1e-4f)
	);
	// (1) multiplier for the single-scatter lobe
	return 1.0f + fAvg * (1.0f - ess) / ess;
}

// compensated directional albedo of a dielectric ggx interface: the
// fresnel-weighted single-scatter albedo times the turquin multiplier
/*
	E(\mu) = (1 + F_{avg} \frac{1 - E_{ss}(\mu)}{E_{ss}(\mu)})
		\cdot E_{ss}^F(\mu, \alpha, f_0)
*/
f32 utilMicrofacetDielectricAlbedo(
	const f32 mu,
	const f32 roughness,
	const f32 f0
) {
	// E_{ss}^F: single-scatter albedo with the actual fresnel
	const f32 essF = (
		utilMicrofacetGgxDirectionalAlbedo(mu, roughness, f0, 1.0f)
	);
	return (
		clamp(
			utilMicrofacetDielectricEnergyCompensate(mu, roughness, f0) * essF,
			0.0f,
			1.0f
		)
	);
}

// the kulla conty energy table as reference. requires the including
// shader to have already declared:
//   layout(set = 0, binding = 0) uniform sampler2D vkofTextures[];
f32 utilMicrofacetDielectricEnergyCompensationKullaConty(
	const u32 kullaContyEnergyHandle,
	const f32 mu,
	const f32 roughness
) {
	// mu axis: (i+0.5f)/16.0f (grazing->normal)
	// roughness axis: (i/16.0f)^2 (smooth->rough)
	//
	// FIXED (diverges from cull): rows are generated at alpha=(i/16)^2 per
	// the comment above, so looking a given alpha up requires the
	// inverse, sqrt(alpha), not alpha directly -- cull's original code
	// used `roughness * 15.0f` here, which is inconsistent with its own
	// documented row spacing. this does NOT make the table converge with
	// the separate analytic (materialx) fit any better -- see
	// unit-test/src/test-energy-compensation.cpp and the artifact report
	// -- but it does make the lookup internally consistent with what the
	// table's own generation comment says it should be, which is the
	// correctness criterion being applied here.
	const f32 ai = clamp(sqrt(roughness) * 15.0f, 0.0f, 14.999f);
	const f32 mi = clamp(mu*16.0f - 0.5f, 0.0f, 14.999f);
	const int ai0 = int(ai);
	const int ai1 = min(ai0 + 1, 15);
	const int mi0 = int(mi);
	const int mi1 = min(mi0 + 1, 15);
	const f32 ta = fract(ai);
	const f32 tm = fract(mi);
	return (
		mix(
			mix(
				texelFetch(
					vkofTextures[kullaContyEnergyHandle], ivec2(mi0, ai0), 0
				).r,
				texelFetch(
					vkofTextures[kullaContyEnergyHandle], ivec2(mi1, ai0), 0
				).r,
				tm
			),
			mix(
				texelFetch(
					vkofTextures[kullaContyEnergyHandle], ivec2(mi0, ai1), 0
				).r,
				texelFetch(
					vkofTextures[kullaContyEnergyHandle], ivec2(mi1, ai1), 0
				).r,
				tm
			),
			ta
		)
	);
}

#endif // UTIL_ENERGY_COMPENSATION_GLSL
