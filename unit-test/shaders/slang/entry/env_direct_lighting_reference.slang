import "util-material-openpbr";
import "util-environment-map";
import "util-env-dl-material";

struct PC {
	u32v4 rootPad[8];
	u64 sumVa;
	EnvironmentMapHandles handles;
	u32 kullaContyEnergyHandle;
	u32 zeltnerLtcParamHandle;
	u32 subGrid;
	f32 rotation;
	f32v3 wi;
	f32v3 baseColor;
	f32 baseMetalness;
	f32 specularRoughness;
	f32 specularIor;
	f32 coatWeight;
	f32 coatRoughness;
	f32 coatIor;
};

[[vk::push_constant]] PC pc;

// deterministic ground truth for env_direct_lighting.comp: a midpoint
// quadrature of
//
//   \int_{S^2} L(\omega) f(\omega_i, \omega) |\cos\theta| d\omega
//
// over the environment map's own uv parameterization, one thread per texel,
// pc.subGrid x pc.subGrid sub-samples inside each texel cell.
//
// L is read through environmentMapRadiance -- the same *bilinear* fetch the
// path tracer uses, not a texel-center point sample. that matters: the
// estimators converge to the integral of the bilinearly reconstructed map,
// which for a high-contrast feature (a sun disc a texel or two wide) is a
// visibly different function from the piecewise-constant texel image the
// pdf table is built from. quoting truth against the same reconstruction
// keeps that a variance question rather than silently folding it into the
// bias the test is trying to measure -- see the sub-texel jitter case for
// the separate defect that turns it into real loss.
//
// d\omega = \sin\theta \, d\theta \, d\phi, and the equirect map spans
// \theta over [0, \pi] in v and \phi over 2\pi in u, so a sub-cell of size
// (1/(W \cdot sub), 1/(H \cdot sub)) in uv carries
// d\omega = (2\pi / (W \cdot sub)) (\pi / (H \cdot sub)) \sin\theta.

[shader("compute")]
[numthreads(64, 1, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const i32v2 size = textureSize(vkofTextures[pc.handles.pdf], 0);
	const u32 texelCount = u32(size.x) * u32(size.y);
	const u32 tid = gl_GlobalInvocationID.x;
	if (tid >= texelCount) {
		return;
	}

	ShadingFrame frame;
	// y-up, matching the environment map's own convention
	// (environmentMapUvToDir returns y = cos(theta)), so v < 0.5 is the
	// surface's upper hemisphere and a "sun" placed high in the map is
	// actually above this surface's horizon
	frame.tanX = f32v3(1.0f, 0.0f, 0.0f);
	frame.tanY = f32v3(0.0f, 0.0f, -1.0f);
	frame.nor = f32v3(0.0f, 1.0f, 0.0f);
	const ShadingFrame coatFrame = frame;

	MaterialTableHandles tables;
	tables.kullaContyEnergy = pc.kullaContyEnergyHandle;
	tables.zeltnerLtcParam = pc.zeltnerLtcParamHandle;

	const OpenPbrMaterial mat = (
		envDlMaterial(
			pc.baseColor, pc.baseMetalness, pc.specularRoughness,
			pc.specularIor, pc.coatWeight, pc.coatRoughness, pc.coatIor
		)
	);
	const f32v3 wi = normalize(pc.wi);
	const f32 dispersedIor = openPbrTransmissionIor(mat);

	const u32 x0 = tid % u32(size.x);
	const u32 y0 = tid / u32(size.x);
	const f32 sub = f32(pc.subGrid);
	const f32 duSub = 1.0f / (f32(size.x) * sub);
	const f32 dvSub = 1.0f / (f32(size.y) * sub);
	// the \sin\theta factor is folded in per sub-sample below
	const f32 solidAngleBase = (TAU * duSub) * (PI * dvSub);

	f32v3 sum = f32v3(0.0f);
	for (u32 j = 0u; j < pc.subGrid; ++j) {
		const f32 v = (f32(y0) + (f32(j) + 0.5f) / sub) / f32(size.y);
		const f32 sinTheta = sin(v * PI);
		for (u32 k = 0u; k < pc.subGrid; ++k) {
			const f32 u = (f32(x0) + (f32(k) + 0.5f) / sub) / f32(size.x);
			const f32v3 dir = environmentMapUvToDir(f32v2(u, v), pc.rotation);
			const f32v3 f = (
				openPbrEvaluateF(
					tables, frame, coatFrame, wi, dir, mat,
					/*isInsideMedium=*/false, dispersedIor
				)
			);
			if (!any((f > f32v3(0.0f)))) {
				continue;
			}
			const f32v3 radiance = (
				environmentMapRadiance(pc.handles, dir, pc.rotation)
			);
			sum += (
				f * abs(dot(frame.nor, dir)) * radiance
				* solidAngleBase * sinTheta
			);
		}
	}

	((f32 *)(pc.sumVa))[tid * 3u + 0u] = sum.x;
	((f32 *)(pc.sumVa))[tid * 3u + 1u] = sum.y;
	((f32 *)(pc.sumVa))[tid * 3u + 2u] = sum.z;
}
