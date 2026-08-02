import "util-global-pc";
import "util-manifold-shade-pc";

struct PC {
	GpuGlobalPc global;
	GpuManifoldShadePC manifoldShade;
};

[[vk::push_constant]] PC pc;

// proof of concept for plan-slang-caustics.md: paints a flat colour wherever
// manifold-probe.slang found a bend point, to show the shape of the caustic
// paths before any brightness is worked out. carries no radiance

// mirrors BendPoint in manifold-probe.slang
struct ManifoldBendPoint {
	f32v3 point;
	u32 converged;
	f32v3 normal;
	u32 pad;
};

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
	const u32v3 gl_GlobalInvocationID : SV_DispatchThreadID,
) {
	const u32v2 coord = gl_GlobalInvocationID.xy;
	if (coord.x >= pc.manifoldShade.width || coord.y >= pc.manifoldShade.height) {
		return;
	}

	ManifoldBendPoint * bendPoints = (
		(ManifoldBendPoint *)(pc.manifoldShade.bendPointsVa)
	);
	const u32 pixel = coord.x + coord.y * pc.manifoldShade.width;
	if (bendPoints[pixel].converged == 0u) { return; }

	const i32v2 texel = i32v2(coord);
	const f32v4 prev = imageLoad(vkofImages[pc.manifoldShade.colorHandle], texel);
	imageStore(
		vkofImages[pc.manifoldShade.colorHandle],
		texel,
		f32v4(prev.rgb + pc.manifoldShade.debugColor, prev.a)
	);
}
