implementing util;

import types;

namespace vkof
{

// a sample of numbers to generate a camera ray
public struct CameraSample {
	// offset sub-pixel sample in [0, 1)
	public f32v2 pixelJitter;
	// time sample in shutter interval [0, 1)
	public f32 shutter;
	// lens sample in unit disk [-1, 1]^2
	public f32v2 lens;
};

// a ray generated from a camera sample
public struct CameraRay {
	public f32v3 origin;
	// normalized
	public f32v3 direction;
	// meters of light from the frame's time origin
	public f32 time;
};


public interface I_Camera {
	public func generateRay(
		pixel: u32v2,
		cameraSample: CameraSample
	) -> CameraRay;
};

// simulates a camera with a pinhole lens
public struct Camera_Pinhole : I_Camera {
	// the view camera
	public __init() {
		this.ori = vkof::global.cameraOri;
		this.forward = vkof::global.cameraForward;
		this.right = vkof::global.cameraRight;
		this.up = vkof::global.cameraUp;
	}

	// right and up scaled like vkof::global.cameraRight and cameraUp
	public __init(f32v3 ori, f32v3 forward, f32v3 right, f32v3 up) {
		this.ori = ori;
		this.forward = forward;
		this.right = right;
		this.up = up;
	}

	public func generateRay(
		pixel: u32v2,
		cameraSample: CameraSample
	) -> CameraRay {
		let size: f32v2 = (
			f32v2(vkof::global.renderWidth, vkof::global.renderHeight)
		);
		let uv: f32v2 = (f32v2(pixel + cameraSample.pixelJitter) / size);
		let ndc: f32v2 = f32v2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
		let direction: f32v3 = (
			normalize(this.forward + ndc.x * this.right + ndc.y * this.up)
		);
		var ray: CameraRay;
		ray.origin = this.ori;
		ray.direction = direction;
		ray.time = (
			vkof::global.cameraShutterOpen
			+ cameraSample.shutter * vkof::global.cameraShutterDuration
		);
		return ray;
	};

	f32v3 ori;
	f32v3 forward;
	f32v3 right;
	f32v3 up;
};

// simulates a camera with a thin lens
public struct Camera_ThinLens : I_Camera {
	Camera_Pinhole pinhole;
	f32 lensRadius;
	f32 focalDistance;

	public __init(Camera_Pinhole pinhole, f32 lensRadius, f32 focalDistance) {
		this.pinhole = pinhole;
		this.lensRadius = lensRadius;
		this.focalDistance = focalDistance;
	}

	public func generateRay(
		pixel: u32v2,
		cameraSample: CameraSample
	) -> CameraRay
	{
		var ray: CameraRay = this.pinhole.generateRay(pixel, cameraSample);
		if (lensRadius <= 0.0f) { return ray; }
		// -- calculate focal point
		// focal point is along the pinhole direction
		let focus: f32v3 = (
			pinhole.ori
			+ ray.direction * (focalDistance / dot(ray.direction, pinhole.forward))
		);
		// -- calculate disk
		let disk: f32v2 = (
			lensRadius * implSampleDiskConcentric(cameraSample.lens)
		);
		// -- calculate new ray
		ray.origin = (
			pinhole.ori
			+ disk.x * normalize(pinhole.right)
			+ disk.y * normalize(pinhole.up)
		);
		ray.direction = normalize(focus - ray.origin);
		return ray;
	}
}

} // namespace vkof

// concentric square-to-disk mapping
func implSampleDiskConcentric(ξ: f32v2) -> f32v2
{
	let o: f32v2 = 2.0f * ξ - 1.0f;
	if (o.x == 0.0f && o.y == 0.0f) { return f32v2(0.0f, 0.0f); }
	let horizontal: bool = abs(o.x) > abs(o.y);
	let r: f32 = horizontal ? o.x : o.y;
	let θ: f32 = (
		horizontal
		? (π / 4.0f) * (o.y / o.x)
		: (π / 2.0f) - (π / 4.0f) * (o.x / o.y)
	);
	return r * f32v2(cos(θ), sin(θ));
}
