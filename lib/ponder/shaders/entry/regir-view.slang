import light;
import medium;
import model;
import path;
import sample;
import types;
import util;
import wavefront;

// debug only: replaces integrate, shading each pixel's primary hit by the
// regir probe its surface nee would use

// mirrors WavefrontRegirViewPc in wavefront.cpp
struct StagePushConstants {
	u32 outputHandle;
	// 1 probe state, 2 a hash color per usable probe
	u32 mode;
	// the display's exposure, divided out so colors survive it
	f32 exposure;
};

[[vk::push_constant]]
ConstantBuffer<vkof::PushConstants<StagePushConstants>> pc;

func linearRec709ToXyz(rgb: f32v3) -> f32v3 {
	return f32v3(
		dot(rgb, f32v3(0.4124564f, 0.3575761f, 0.1804375f)),
		dot(rgb, f32v3(0.2126729f, 0.7151522f, 0.0721750f)),
		dot(rgb, f32v3(0.0193339f, 0.1191920f, 0.9503041f))
	);
}

func hashColor(slot: u32) -> f32v3 {
	var h: u32 = slot * 747796405u + 2891336453u;
	h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
	h = (h >> 22u) ^ h;
	return (
		f32v3(f32(h & 0xFFu), f32((h >> 8u) & 0xFFu), f32((h >> 16u) & 0xFFu))
		/ 255.0f * 0.8f + 0.2f
	);
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
	let sampler: vkof::Sampler = vkof::Sampler(threadId.xy, 0u);
	let cameraPinhole: vkof::Camera_Pinhole = vkof::Camera_Pinhole();
	let camera: vkof::Camera_ThinLens = (
		vkof::Camera_ThinLens(
			/*pinhole=*/ cameraPinhole,
			/*lensRadius=*/ vkof::global.cameraLensRadius,
			/*focalDistance=*/ vkof::global.cameraFocalDistance,
		)
	);
	let cameraSample: vkof::CameraSample = {
		/*pixelJitter=*/ f32v2(0.5f),
		/*shutter=*/ 0.0f,
		/*lens=*/ sampler.nextF32v2(vkof::SampleDimension.CameraLens),
	};
	let cameraRay: vkof::CameraRay = (
		camera.generateRay(/*pixel=*/ threadId.xy, /*sample=*/ cameraSample)
	);
	let state: vkof::WavefrontTransportState = {
		/*ori=*/ cameraRay.origin,
		/*dir=*/ cameraRay.direction,
		/*λ=*/ vkof::wavelengthSampleUniform(0.5f),
		/*cameraTime=*/ cameraRay.time,
		/*pathLength=*/ 0.0f,
		/*mediumStack=*/ vkof::MediumStack(),
	};
	vkof::I_TraceTransport transport = vkof::TraceTransport_Bend(state);
	var stream: vkof::SampleStream = (
		sampler.bounceView(0u).stream(vkof::BounceDimension.MediumDistance)
	);
	let hit: vkof::TraceTransportHit = (
		transport.traceTransport(
			/*maxDistance=*/ vkof::skEnvironmentDistance,
			/*stream=*/ stream
		)
	);

	var rgb: f32v3 = f32v3(0.02f);
	if (hit.event == vkof::TransportEvent::Surface) {
		var geometry: vkof::SurfaceGeometry = (
			vkof::surfaceGeometryFromHit(
				/*hit=*/ hit.surface.hit,
				/*worldPosition=*/ hit.surface.position
			)
		);
		// -- the face-forwarded normal, as the surface nee site keys with
		if (dot(geometry.geometricNormal, hit.incomingDirection) > 0.0f) {
			geometry.geometricNormal = -geometry.geometricNormal;
		}
		let normal: Optional<f32v3> = Optional<f32v3>(geometry.geometricNormal);
		let key: vkof::RegirCellKey = vkof::regirKeyFor(hit.position, normal);
		var slot: u32 = vkof::regirLookup(key);
		// -- claim like the surface nee site, so the view keeps the grid live
		if (slot == vkof::skRegirSlotInvalid) {
			slot = vkof::regirFindOrInsert(key, hit.position, normal);
		}
		if (slot == vkof::skRegirSlotInvalid) {
			rgb = f32v3(0.8f, 0.05f, 0.05f);
		} else {
			vkof::regirRecordShadingPoint(slot, hit.position, geometry.geometricNormal);
			let cell: vkof::RegirCell = vkof::global.regirCells[slot];
			if (cell.lastFillIndex == vkof::skRegirSlotInvalid) {
				rgb = f32v3(0.8f, 0.7f, 0.05f);
			} else if (cell.lastFillIndex != vkof::global.regirFillIndex) {
				rgb = f32v3(0.3f);
			} else if (!(cell.normalizationNoncanonical > 0.0f)) {
				rgb = f32v3(0.05f, 0.25f, 0.05f);
			} else {
				rgb = (
					pc.stage.mode == 2u
					? hashColor(slot)
					: f32v3(0.1f, 0.75f, 0.2f)
				);
			}
		}
		// -- a little shading so the geometry stays readable
		let facing: f32 = abs(dot(geometry.geometricNormal, hit.incomingDirection));
		rgb *= 0.35f + 0.65f * facing;
	}
	vkofImages[pc.stage.outputHandle][threadId.xy] = (
		f32v4(linearRec709ToXyz(rgb / max(pc.stage.exposure, 1e-6f)), 1.0f)
	);
}
