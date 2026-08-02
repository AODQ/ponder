import types;
import util;

// -----------------------------------------------------------------------------
// DLSS ray reconstruction inputs from a finished pass: linear rec.709 color,
// and view depth and camera motion vectors from the primary hit position

// mirrors DlssRrPreparePc in lib/ponder/src/dlss-rr.cpp
struct StagePushConstants {
	// the pass's xyz radiance
	u32 colorHandle;
	// the primary hit position aov, w = 1 on a surface
	u32 positionHandle;
	u32 rrColorHandle;
	u32 depthHandle;
	u32 motionHandle;
	// the previous frame's camera, scaled like vkof::global.camera*
	f32v3 previousOri;
	f32v3 previousForward;
	f32v3 previousRight;
	f32v3 previousUp;
};

[[vk::push_constant]]
ConstantBuffer<vkof::PushConstants<StagePushConstants>> pc;

// past any scene, and within f16 range
static let skSkyDepth: f32 = 65504.0f;

// render pixel coordinates of a camera-space direction
func projectPixel(
	direction: f32v3,
	forward: f32v3,
	right: f32v3,
	up: f32v3
) -> Optional<f32v2> {
	let z: f32 = dot(direction, forward);
	if (z <= 0.0f) { return none; }
	let ndc: f32v2 = f32v2(
		dot(direction, right) / (dot(right, right) * z),
		dot(direction, up) / (dot(up, up) * z)
	);
	let size: f32v2 = (
		f32v2(vkof::global.renderWidth, vkof::global.renderHeight)
	);
	return f32v2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f) * size;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
	const u32v3 threadId : SV_DispatchThreadID,
) {
	vkof::globalBind(pc.global);
	if (
		threadId.x >= vkof::global.renderWidth
		|| threadId.y >= vkof::global.renderHeight
	) {
		return;
	}
	let coord: u32v2 = threadId.xy;
	let stage: StagePushConstants = pc.stage;

	// -- color; a non-finite sample would poison the history
	let xyz: f32v3 = vkofImages[stage.colorHandle][coord].xyz;
	let rgb: f32v3 = (
		all(isfinite(xyz))
		? max(vkof::xyzToLinearRec709(xyz), 0.0f)
		: f32v3(0.0f)
	);
	vkofImages[stage.rrColorHandle][coord] = f32v4(rgb, 1.0f);

	// -- depth and motion; the sky moves with the camera rotation only
	let position: f32v4 = vkofImages[stage.positionHandle][coord];
	let cameraOri: f32v3 = vkof::global.cameraOri;
	let cameraForward: f32v3 = vkof::global.cameraForward;
	let cameraRight: f32v3 = vkof::global.cameraRight;
	let cameraUp: f32v3 = vkof::global.cameraUp;
	let onSurface: bool = position.w > 0.0f;
	let pixelCenter: f32v2 = f32v2(coord) + 0.5f;
	let size: f32v2 = (
		f32v2(vkof::global.renderWidth, vkof::global.renderHeight)
	);
	let ndc: f32v2 = f32v2(
		pixelCenter.x / size.x * 2.0f - 1.0f,
		1.0f - pixelCenter.y / size.y * 2.0f
	);
	let skyDirection: f32v3 = (
		cameraForward + ndc.x * cameraRight + ndc.y * cameraUp
	);
	var current: Optional<f32v2> = pixelCenter;
	if (onSurface) {
		current = projectPixel(
			position.xyz - cameraOri, cameraForward, cameraRight, cameraUp
		);
	}
	let previous: Optional<f32v2> = (
		projectPixel(
			onSurface ? position.xyz - stage.previousOri : skyDirection,
			stage.previousForward,
			stage.previousRight,
			stage.previousUp
		)
	);
	let motion: f32v2 = (
		current.hasValue && previous.hasValue
		? previous.value - current.value
		: f32v2(0.0f)
	);
	let depth: f32 = (
		onSurface ? dot(position.xyz - cameraOri, cameraForward) : skSkyDepth
	);
	vkofImages[stage.depthHandle][coord] = f32v4(depth, 0.0f, 0.0f, 0.0f);
	vkofImages[stage.motionHandle][coord] = f32v4(motion, 0.0f, 0.0f);
}
