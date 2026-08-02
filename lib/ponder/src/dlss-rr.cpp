#include <ponder/dlss-rr.hpp>

#if defined(PONDER_DLSS)
#include <vkof/ngx.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace {

static constexpr char const * skPonderShaderNewDir = PONDER_SHADER_NEW_DIR;

// rr recommends at least 32 jitter phases, and 8 per output pixel a render
// pixel covers when upscaling
static constexpr u32 skJitterPhaseCountMin = 64u;

// mirrors StagePushConstants in entry/dlss-rr-prepare.slang
struct DlssRrPreparePc {
	u32 colorHandle;
	u32 positionHandle;
	u32 rrColorHandle;
	u32 depthHandle;
	u32 motionHandle;
	f32v3 previousOri;
	f32v3 previousForward;
	f32v3 previousRight;
	f32v3 previousUp;
};
static_assert(sizeof(DlssRrPreparePc) == 68u);

// mirrors StagePushConstants in entry/dlss-rr-resolve.slang
struct DlssRrResolvePc {
	u32 rrOutputHandle;
	u32 resolvedHandle;
	u32 outputHandle;
	f32 exposure;
	u32 outputWidth;
	u32 outputHeight;
};
static_assert(sizeof(DlssRrResolvePc) == 24u);

f32 halton(u32 index, u32 const base) {
	f32 result = 0.0f;
	f32 fraction = 1.0f;
	while (index > 0u) {
		fraction /= (f32)base;
		result += fraction * (f32)(index % base);
		index /= base;
	}
	return result;
}

u32 jitter_phase_count(ponder::DlssRr const & rr) {
	f32 const ratio = (
		(f32)(rr.outputSize.x * rr.outputSize.y)
		/ (f32)std::max(rr.renderSize.x * rr.renderSize.y, 1u)
	);
	return std::max(skJitterPhaseCountMin, (u32)std::ceil(8.0f * ratio));
}

#if defined(PONDER_DLSS)
// rr's projection only maps view x and y; depth is passed linear
static constexpr f32 skProjectionNear = 0.01f;
static constexpr f32 skProjectionFar = 1.0e4f;

// rr's camera matrices: row-major, row vectors (v * m), d3d clip space, no
// jitter; view space is x right, y up, z forward like the depth guide
void camera_matrices(
	ponder::WavefrontCamera const & camera,
	f32 const aspect,
	f32 (& worldToView)[16],
	f32 (& viewToClip)[16]
) {
	ponder::WavefrontCameraBasis const basis = (
		ponder::wavefront_camera_basis(camera, aspect)
	);
	f32v3 const right = f32v3_normalize(basis.right);
	f32v3 const up = f32v3_normalize(basis.up);
	f32v3 const forward = basis.forward;
	f32v3 const o = camera.position;
	f32 const view[16] = {
		right.x, up.x, forward.x, 0.0f,
		right.y, up.y, forward.y, 0.0f,
		right.z, up.z, forward.z, 0.0f,
		-f32v3_dot(o, right), -f32v3_dot(o, up), -f32v3_dot(o, forward), 1.0f,
	};
	f32 const tanY = f32v3_length(basis.up);
	f32 const tanX = f32v3_length(basis.right);
	f32 const zScale = skProjectionFar / (skProjectionFar - skProjectionNear);
	f32 const clip[16] = {
		1.0f / tanX, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f / tanY, 0.0f, 0.0f,
		0.0f, 0.0f, zScale, 1.0f,
		0.0f, 0.0f, -skProjectionNear * zScale, 0.0f,
	};
	std::copy(view, view + 16, worldToView);
	std::copy(clip, clip + 16, viewToClip);
}
#endif

struct ImageWithHandle {
	vkof::Image image;
	u32 handle;
};

ImageWithHandle image_create(
	u32 const width,
	u32 const height,
	vkof::ImageFormat const format
) {
	vkof::Image const image = (
		vkof::image_create({
			.width = width,
			.height = height,
			.depth = 1u,
			.format = format,
			.mipLevels = 1u,
			.optInitialData = {},
		})
	);
	return ImageWithHandle {
		.image = image,
		.handle = vkof::image_storage_handle({ .image = image, .mipLevel = 0u }),
	};
}

void images_destroy(ponder::DlssRr & rr) {
	for (vkof::Image const image : {
		rr.colorImage,
		rr.diffuseAlbedoImage,
		rr.specularAlbedoImage,
		rr.roughnessImage,
		rr.depthImage,
		rr.motionImage,
		rr.specularHitDistanceImage,
		rr.positionImage,
		rr.normalImage,
		rr.outputImage,
	}) {
		vkof::image_destroy(image);
	}
}

void images_create(
	ponder::DlssRr & rr,
	u32v2 const renderSize,
	u32v2 const outputSize
) {
	using vkof::ImageFormat;
	u32 const width = renderSize.x;
	u32 const height = renderSize.y;
	auto const color = image_create(width, height, ImageFormat::r16g16b16a16_sfloat);
	auto const diffuseAlbedo = (
		image_create(width, height, ImageFormat::r16g16b16a16_sfloat)
	);
	auto const specularAlbedo = (
		image_create(width, height, ImageFormat::r16g16b16a16_sfloat)
	);
	auto const roughness = image_create(width, height, ImageFormat::r32_float);
	auto const depth = image_create(width, height, ImageFormat::r32_float);
	auto const specularHitDistance = (
		image_create(width, height, ImageFormat::r32_float)
	);
	auto const position = (
		image_create(width, height, ImageFormat::r32g32b32a32_sfloat)
	);
	auto const normal = (
		image_create(width, height, ImageFormat::r16g16b16a16_sfloat)
	);
	auto const motion = (
		image_create(width, height, ImageFormat::r16g16b16a16_sfloat)
	);
	auto const output = (
		image_create(
			outputSize.x, outputSize.y, ImageFormat::r16g16b16a16_sfloat
		)
	);
	rr.colorImage = color.image;
	rr.colorHandle = color.handle;
	rr.diffuseAlbedoImage = diffuseAlbedo.image;
	rr.diffuseAlbedoHandle = diffuseAlbedo.handle;
	rr.specularAlbedoImage = specularAlbedo.image;
	rr.specularAlbedoHandle = specularAlbedo.handle;
	rr.roughnessImage = roughness.image;
	rr.roughnessHandle = roughness.handle;
	rr.depthImage = depth.image;
	rr.depthHandle = depth.handle;
	rr.motionImage = motion.image;
	rr.motionHandle = motion.handle;
	rr.specularHitDistanceImage = specularHitDistance.image;
	rr.specularHitDistanceHandle = specularHitDistance.handle;
	rr.positionImage = position.image;
	rr.positionHandle = position.handle;
	rr.normalImage = normal.image;
	rr.normalHandle = normal.handle;
	rr.outputImage = output.image;
	rr.outputHandle = output.handle;
	rr.renderSize = renderSize;
	rr.outputSize = outputSize;
}

} // namespace

bool ponder::dlss_rr_supported() {
#if defined(PONDER_DLSS)
	return vkof::ngx_rr_available();
#else
	return false;
#endif
}

char const * ponder::dlss_rr_unsupported_reason() {
#if defined(PONDER_DLSS)
	return vkof::ngx_rr_unavailable_reason();
#else
	return "built without PONDER_DLSS";
#endif
}

u32v2 ponder::dlss_rr_render_size(
	DlssRrQuality const quality,
	u32v2 const outputSize
) {
#if defined(PONDER_DLSS)
	return vkof::ngx_rr_render_size((vkof::NgxRrQuality)quality, outputSize);
#else
	(void)quality;
	return outputSize;
#endif
}

void ponder::dlss_rr_prepare(
	DlssRr & rr,
	u32v2 const renderSize,
	u32v2 const outputSize
) {
	if (!rr.created) {
		char const * const includePaths[] = { skPonderShaderNewDir };
		std::string const preparePath = (
			std::string(skPonderShaderNewDir) + "entry/dlss-rr-prepare.slang"
		);
		std::string const resolvePath = (
			std::string(skPonderShaderNewDir) + "entry/dlss-rr-resolve.slang"
		);
		vkof::PipelineComputeCreateInfo const createInfos[] = {
			vkof::PipelineComputeCreateInfo {
				.pathCompute = preparePath.c_str(),
				.includePaths = srat::slice { includePaths, 1u },
			},
			vkof::PipelineComputeCreateInfo {
				.pathCompute = resolvePath.c_str(),
				.includePaths = srat::slice { includePaths, 1u },
			},
		};
		vkof::Pipeline pipelines[2] {};
		vkof::pipeline_compute_create_batch(
			srat::slice { createInfos, 2u },
			srat::slice { pipelines, 2u }
		);
		rr.preparePipeline = pipelines[0];
		rr.resolvePipeline = pipelines[1];
		images_create(rr, renderSize, outputSize);
		rr.created = true;
		rr.hasPrevious = false;
		return;
	}
	if (rr.renderSize != renderSize || rr.outputSize != outputSize) {
		vkof::device_wait_idle();
		images_destroy(rr);
		images_create(rr, renderSize, outputSize);
		rr.hasPrevious = false;
	}
}

void ponder::dlss_rr_destroy(DlssRr & rr) {
	if (!rr.created) { return; }
#if defined(PONDER_DLSS)
	vkof::ngx_rr_release();
#endif
	vkof::pipeline_destroy(rr.preparePipeline);
	vkof::pipeline_destroy(rr.resolvePipeline);
	images_destroy(rr);
	rr = {};
}

f32v2 ponder::dlss_rr_jitter(DlssRr const & rr) {
	// halton index 0 is the pixel corner; start at 1
	u32 const index = rr.jitterPhase % jitter_phase_count(rr) + 1u;
	return f32v2 { halton(index, 2u), halton(index, 3u) };
}

ponder::WavefrontRrGuide ponder::dlss_rr_guide(DlssRr const & rr) {
	return WavefrontRrGuide {
		.diffuseAlbedoImage = rr.diffuseAlbedoImage,
		.diffuseAlbedoHandle = rr.diffuseAlbedoHandle,
		.specularAlbedoImage = rr.specularAlbedoImage,
		.specularAlbedoHandle = rr.specularAlbedoHandle,
		.roughnessImage = rr.roughnessImage,
		.roughnessHandle = rr.roughnessHandle,
		.specularHitDistanceImage = rr.specularHitDistanceImage,
		.specularHitDistanceHandle = rr.specularHitDistanceHandle,
		.positionImage = rr.positionImage,
		.positionHandle = rr.positionHandle,
		.normalImage = rr.normalImage,
		.normalHandle = rr.normalHandle,
	};
}

ponder::DlssRrNodes ponder::dlss_rr_nodes(
	DlssRr & rr,
	DlssRrFrameInfo const & info
) {
	u32 const width = rr.renderSize.x;
	u32 const height = rr.renderSize.y;
	u32 const outputWidth = rr.outputSize.x;
	u32 const outputHeight = rr.outputSize.y;
	f32 const aspect = (f32)width / (f32)height;
	WavefrontCamera const previousCamera = (
		rr.hasPrevious ? rr.previousCamera : info.camera
	);
	WavefrontCameraBasis const previous = (
		wavefront_camera_basis(previousCamera, aspect)
	);
	bool const reset = info.reset || !rr.hasPrevious;

	// -- prepare: rr color, depth and motion
	vkof::RenderNode const prepare = (
		vkof::render_node_create({
			.queue = vkof::CommandQueue::graphics,
			.debugName = "dlss-rr prepare",
		})
	);
	vkof::render_node_add_image({
		.node = prepare,
		.image = info.colorImage,
		.access = vkof::RenderNodeAccess::read,
	});
	vkof::render_node_add_persistent_image({
		.node = prepare,
		.image = rr.positionImage,
		.access = vkof::RenderNodeAccess::read,
	});
	for (vkof::Image const image : {
		rr.colorImage, rr.depthImage, rr.motionImage
	}) {
		vkof::render_node_add_persistent_image({
			.node = prepare,
			.image = image,
			.access = vkof::RenderNodeAccess::write,
		});
	}
	auto const preparePc = DlssRrPreparePc {
		.colorHandle = info.colorHandle,
		.positionHandle = rr.positionHandle,
		.rrColorHandle = rr.colorHandle,
		.depthHandle = rr.depthHandle,
		.motionHandle = rr.motionHandle,
		.previousOri = previousCamera.position,
		.previousForward = previous.forward,
		.previousRight = previous.right,
		.previousUp = previous.up,
	};
	vkof::Pipeline const preparePipeline = rr.preparePipeline;
	vkof::render_node_callback({
		.node = prepare,
		.callback = [=](vkof::CommandBuffer const & cmd) {
			vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
				.cmd = cmd,
				.pipeline = preparePipeline,
				.push = preparePc,
				.threadgroupSize = u32v3 { 16u, 16u, 1u },
				.invocationCount = u32v3 { width, height, 1u },
			});
		},
	});

	// -- evaluate
	vkof::RenderNode const evaluate = (
		vkof::render_node_create({
			.queue = vkof::CommandQueue::graphics,
			.debugName = "dlss-rr evaluate",
		})
	);
	for (vkof::Image const image : {
		rr.colorImage,
		rr.diffuseAlbedoImage,
		rr.specularAlbedoImage,
		rr.normalImage,
		rr.roughnessImage,
		rr.depthImage,
		rr.motionImage,
		rr.specularHitDistanceImage,
	}) {
		vkof::render_node_add_persistent_image({
			.node = evaluate,
			.image = image,
			.access = vkof::RenderNodeAccess::read,
		});
	}
	vkof::render_node_add_persistent_image({
		.node = evaluate,
		.image = rr.outputImage,
		.access = vkof::RenderNodeAccess::write,
	});
#if defined(PONDER_DLSS)
	f32v2 const jitter = dlss_rr_jitter(rr);
	auto evaluateInfo = vkof::NgxRrEvaluateInfo {
		.cmd = {},
		.renderSize = rr.renderSize,
		.outputSize = rr.outputSize,
		.quality = (vkof::NgxRrQuality)info.quality,
		.preset = (vkof::NgxRrPreset)info.preset,
		.color = rr.colorImage,
		.diffuseAlbedo = rr.diffuseAlbedoImage,
		.specularAlbedo = rr.specularAlbedoImage,
		.normals = rr.normalImage,
		.roughness = rr.roughnessImage,
		.depth = rr.depthImage,
		.motionVectors = rr.motionImage,
		.specularHitDistance = rr.specularHitDistanceImage,
		.worldToView = {},
		.viewToClip = {},
		.output = rr.outputImage,
		// the pixel center's offset from the sample; the opposite sign measured
		// worse against an accumulated reference on both axes
		.jitter = f32v2 { 0.5f - jitter.x, 0.5f - jitter.y },
		.reset = reset,
	};
	camera_matrices(
		info.camera, aspect, evaluateInfo.worldToView, evaluateInfo.viewToClip
	);
	bool * const evaluateFailed = &rr.evaluateFailed;
	vkof::render_node_callback({
		.node = evaluate,
		.callback = [=](vkof::CommandBuffer const & cmd) {
			auto evaluateCmd = evaluateInfo;
			evaluateCmd.cmd = cmd;
			*evaluateFailed = !vkof::ngx_rr_evaluate(evaluateCmd);
		},
	});
#else
	(void)reset;
#endif

	// -- resolve: linear rec.709 and display
	vkof::RenderNode const resolve = (
		vkof::render_node_create({
			.queue = vkof::CommandQueue::graphics,
			.debugName = "dlss-rr resolve",
		})
	);
	vkof::render_node_add_persistent_image({
		.node = resolve,
		.image = rr.outputImage,
		.access = vkof::RenderNodeAccess::read,
	});
	vkof::render_node_add_persistent_image({
		.node = resolve,
		.image = info.resolvedImage,
		.access = vkof::RenderNodeAccess::write,
	});
	vkof::render_node_add_image({
		.node = resolve,
		.image = info.outputImage,
		.access = vkof::RenderNodeAccess::write,
	});
	auto const resolvePc = DlssRrResolvePc {
		.rrOutputHandle = rr.outputHandle,
		.resolvedHandle = info.resolvedHandle,
		.outputHandle = info.outputHandle,
		.exposure = info.exposure,
		.outputWidth = outputWidth,
		.outputHeight = outputHeight,
	};
	vkof::Pipeline const resolvePipeline = rr.resolvePipeline;
	vkof::render_node_callback({
		.node = resolve,
		.callback = [=](vkof::CommandBuffer const & cmd) {
			vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
				.cmd = cmd,
				.pipeline = resolvePipeline,
				.push = resolvePc,
				.threadgroupSize = u32v3 { 16u, 16u, 1u },
				.invocationCount = u32v3 { outputWidth, outputHeight, 1u },
			});
		},
	});

	// -- next frame
	++rr.jitterPhase;
	rr.previousCamera = info.camera;
	rr.hasPrevious = true;
	return DlssRrNodes {
		.prepare = prepare,
		.evaluate = evaluate,
		.resolve = resolve,
	};
}
