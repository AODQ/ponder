import medium;
import path;
import sample;
import types;
import util;
import wavefront;

// mirrors WavefrontCameraPc in lib/ponder/src/wavefront.cpp
struct StagePushConstants {
	// one offset in [0, 1) for every pixel, replacing the per-sample
	// PixelJitter when globalJitterEnabled != 0 (DLSS ray reconstruction)
	f32v2 globalJitter;
	u32 globalJitterEnabled;
};

[[vk::push_constant]]
ConstantBuffer<vkof::PushConstants<StagePushConstants>> pc;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(
	const u32v3 threadId : SV_DispatchThreadID,
) {
	vkof::globalBind(pc.global);
	let slot: Optional<vkof::WavefrontPath> = vkof::wavefrontPath(threadId.x);
	if (!slot.hasValue) { return; }
	let pathIndex: u32 = slot.value.index;
	let pixel: u32v2 = slot.value.pixel;
	vkof::debugPathBegin(pixel);

	// -- setup sampler
	let sampler: vkof::Sampler = (
		vkof::Sampler(pixel, slot.value.sampleIndex*4)
	);

	// -- generate camera ray
	// the debug pixel traces from the view it was picked from
	let cameraPinhole: vkof::Camera_Pinhole = (
		vkof::debugPathActive()
		? vkof::Camera_Pinhole(
			vkof::global.debugCameraOri,
			vkof::global.debugCameraForward,
			vkof::global.debugCameraRight,
			vkof::global.debugCameraUp
		)
		: vkof::Camera_Pinhole()
	);
	let camera: vkof::Camera_ThinLens = (
		vkof::Camera_ThinLens(
			/*pinhole=*/ cameraPinhole,
			/*lensRadius=*/ vkof::global.cameraLensRadius,
			/*focalDistance=*/ vkof::global.cameraFocalDistance,
		)
	);
	var pixelJitter: f32v2 = pc.stage.globalJitter;
	if (pc.stage.globalJitterEnabled == 0u) {
		pixelJitter = sampler.nextF32v2(vkof::SampleDimension.PixelJitter);
	}
	let cameraSample: vkof::CameraSample = {
		/*pixelJitter=*/ pixelJitter,
		/*shutter*/ sampler.nextF32(vkof::SampleDimension.CameraShutter),
		/*lens=*/ sampler.nextF32v2(vkof::SampleDimension.CameraLens),
	};
	let cameraRay: vkof::CameraRay = (
		camera.generateRay(
			/*pixel=*/ pixel,
			/*sample=*/ cameraSample,
		)
	);

	// -- write out the state
	let state: vkof::WavefrontTransportState = {
		/*ori=*/ cameraRay.origin,
		/*dir=*/ cameraRay.direction,
		/*λ=*/ vkof::wavelengthSampleUniform(
			sampler.nextF32(vkof::SampleDimension.Wavelength)
		),
		/*cameraTime=*/ cameraRay.time,
		/*pathLength=*/ 0.0f,
		/*mediumStack=*/ vkof::MediumStack(),
	};
	vkof::wavefrontTransportStateStore(pathIndex, state);
	vkof::pathStateStore(pathIndex, vkof::PathState(state.λ));
	vkof::debugPathPoint(cameraRay.origin, vkof::skDebugPathPointChord);
}
