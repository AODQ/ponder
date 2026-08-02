import path;
import sample;
import types;
import util;
import wavefront;

// shades one bounce of every live path
// mirrors WavefrontIntegratePc in lib/ponder/src/wavefront.cpp

struct StagePushConstants {
	u32 outputHandle;
	// the bounce this dispatch samples, < skPathBounceMax
	u32 bounce;
	// 0 skips the aov images
	u32 aovEnabled;
	u32 aovPositionHandle;
	u32 aovNormalHandle;
	u32 aovAlbedoHandle;
	// 0 skips the DLSS ray reconstruction guide images
	u32 rrGuideEnabled;
	u32 rrDiffuseAlbedoHandle;
	u32 rrSpecularAlbedoHandle;
	u32 rrRoughnessHandle;
	u32 rrSpecularHitDistanceHandle;
	u32 rrPositionHandle;
	u32 rrNormalHandle;
};

[[vk::push_constant]]
ConstantBuffer<vkof::PushConstants<StagePushConstants>> pc;

// -----------------------------------------------------------------------------
// -- one-shot tests
// -----------------------------------------------------------------------------

func hueToRgb(hue: f32) -> f32v3 {
	let k: f32v3 = frac(hue + f32v3(0.0f, 2.0f / 3.0f, 1.0f / 3.0f));
	return saturate(abs(k * 6.0f - 3.0f) - 1.0f);
}

func testSpectrum(threadId: u32v3) -> void {
	u32 width;
	u32 height;
	vkofImages[pc.stage.outputHandle].GetDimensions(width, height);
	let hue: f32 = f32(threadId.x) / f32(width);
	// saturation 0.75, value 0.8, stays inside the sRGB gamut
	let saturation: f32 = 0.65f;
	let value: f32 = 0.8f;
	let rgb: f32v3 = lerp(f32v3(value), hueToRgb(hue) * value, saturation);

	// average 64 stratified wavelength sets lit by d65
	var spectral: f32v3 = f32v3(0.0f);
	let setCount: u32 = 32u;
	for (u32 k = 0u; k < setCount; ++k) {
		let normalizedSample: f32 = ((f32)k + 0.5f) / ((f32)setCount);
		let λ: vkof::Wavelengths = (
			vkof::wavelengthSampleUniform(normalizedSample)
		);
		var reflectance: vkof::SpectrumSampled = (
			vkof::SpectrumSampled.fromRgb(rgb, λ)
		);
		[ForceUnroll]
		for (u32 i = 0u; i < 4u; ++i) {
			reflectance.amounts[i] *= vkof::cieD65(λ.lambdas[i]);
		}
		spectral += reflectance.asRgb(λ);
	}
	spectral /= f32(setCount);

	let band: u32 = threadId.y * 3u / height;
	let out: f32v3 = (
		band == 0u
		? rgb
		: band == 1u
		? spectral
		: abs(spectral - rgb) * 10.5f
	);
	vkofImages[pc.stage.outputHandle][u32v2(threadId.xy)] = f32v4(out, 1.0f);
}

// -----------------------------------------------------------------------------
// -- entry point
// -----------------------------------------------------------------------------

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
	vkof::debugPathResume(pixel);
	// -- one shot tests
	// testSpectrum(threadId);

	// -- setup sampler
	let sampler: vkof::Sampler = (
		vkof::Sampler(pixel, slot.value.sampleIndex*4)
	);

	// -- load the path state and transport state
	var path: vkof::PathState = vkof::pathStateLoad(pathIndex);
	if (!path.alive) { return; }
	let state: vkof::WavefrontTransportState = (
		vkof::wavefrontTransportStateLoad(pathIndex)
	);

	// -- setup transport
	vkof::I_TraceTransport transport = vkof::TraceTransport_Bend(state);
	let hitRecord: vkof::PathHitRecord = (
		vkof::pathHitRecordLoad(pathIndex)
	);

	// -- primary hit aovs
	if (pc.stage.aovEnabled != 0u && pc.stage.bounce == 0u) {
		let aov: vkof::PathAov = vkof::pathAovFromHit(hitRecord.hit);
		vkofImages[pc.stage.aovPositionHandle][pixel] = aov.position;
		vkofImages[pc.stage.aovNormalHandle][pixel] = (
			f32v4(aov.shadingNormal, 0.0f)
		);
		vkofImages[pc.stage.aovAlbedoHandle][pixel] = (
			f32v4(aov.albedo, 1.0f)
		);
	}

	// -- primary hit ray reconstruction guides
	if (pc.stage.rrGuideEnabled != 0u && pc.stage.bounce == 0u) {
		let guide: vkof::PathRrGuide = vkof::pathRrGuideFromHit(hitRecord.hit);
		vkofImages[pc.stage.rrDiffuseAlbedoHandle][pixel] = (
			f32v4(guide.diffuseAlbedo, 1.0f)
		);
		vkofImages[pc.stage.rrSpecularAlbedoHandle][pixel] = (
			f32v4(guide.specularAlbedo, 1.0f)
		);
		vkofImages[pc.stage.rrRoughnessHandle][pixel] = (
			f32v4(guide.roughness, 0.0f, 0.0f, 0.0f)
		);
		vkofImages[pc.stage.rrSpecularHitDistanceHandle][pixel] = (
			f32v4(guide.specularHitDistance, 0.0f, 0.0f, 0.0f)
		);
		vkofImages[pc.stage.rrPositionHandle][pixel] = guide.position;
		vkofImages[pc.stage.rrNormalHandle][pixel] = f32v4(guide.normal, 0.0f);
	}

	// -- evaluate the path radiance and shade
#if defined(PONDER_DEBUG_PATH)
	typealias Consumer =
		vkof::PathConsumer_DebugRecord<vkof::PathConsumer_Evaluate>;
	var consumer: Consumer = Consumer(path.consumer);
#else
	var consumer: vkof::PathConsumer_Evaluate = path.consumer;
#endif
	let neeRecord: vkof::PathNeeRecord = (
		path.shade(
			/*transport=*/ transport,
			/*record=*/ hitRecord,
			/*sampler=*/ sampler,
			/*bounce=*/ pc.stage.bounce,
			/*cameraTime=*/ state.cameraTime,
			/*consumer=*/ consumer
		)
	);
#if defined(PONDER_DEBUG_PATH)
	path.consumer = consumer.innerConsumer;
#else
	path.consumer = consumer;
#endif

	// -- kill the path if it reached the max bounce
	if (pc.stage.bounce + 1u >= vkof::skPathBounceMax) {
		path.alive = false;
	}

	// -- store the path state and transport state
	vkof::wavefrontTransportStateStore(pathIndex, transport.state());
	vkof::pathStateStore(pathIndex, path);
	vkof::pathNeeRecordStore(pathIndex, neeRecord);
}
