#include <ponder/wavefront.hpp>

#include <ponder/debug-path.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

static constexpr char const * skPonderShaderNewDir = PONDER_SHADER_NEW_DIR;
static constexpr char const * skPonderAssetDir = PONDER_ASSET_DIR;
static constexpr f32v3 skWorldUp = { 0.0f, 1.0f, 0.0f };

// mirrors Global in types/global.slang
struct GpuGlobal {
	u64 rgb2specVa;
	u32 rgb2specResolution;
	u32 frameIndex;
	u64 modelsVa;
	u32 renderWidth;
	u32 renderHeight;
	f32v3 cameraOri;
	f32 shutterOpen;
	f32v3 cameraForward;
	f32 shutterDuration;
	f32v3 cameraRight;
	u32 sampleIndex;
	f32v3 cameraUp;
	f32 blackholeMass;
	f32v3 blackholeOri;
	f32 blackholeLensRadius;
	u32 blackholeMaxChords;
	f32 blackholeDiskInnerRadius;
	f32 blackholeDiskOuterRadius;
	f32 blackholeDiskTemperature;
	f32 blackholeDiskBrightness;
	f32 blackholeDiskOpticalDepth;
	f32 blackholeDiskOpticalDepthExponent;
	f32 blackholeStepFactor;
	f32 blackholeDiskTime;
	f32 blackholeDiskRotation;
	f32 blackholeDiskNoiseAmplitude;
	f32 blackholeDiskNoiseScale;
	f32 blackholeDiskHotspotRadius;
	f32 blackholeDiskHotspotTemperature;
	f32 blackholeDiskHotspotSize;
	f32 blackholeSpin;
	u64 debugPathVa;
	u32v2 debugPixel;
	f32v3 debugCameraOri;
	u32 pad1;
	f32v3 debugCameraForward;
	u32 pad2;
	f32v3 debugCameraRight;
	u32 pad3;
	f32v3 debugCameraUp;
	u32 pad4;
	f32v3 ambientMediumSigmaAbsorption;
	u32 ambientMediumType;
	f32v3 ambientMediumSigmaScattering;
	u32 ambientMediumPhase;
	f32 ambientMediumAnisotropy;
	f32 ambientMediumDraineAlpha;
	f32 ambientMediumDropletDiameter;
	f32 ambientMediumHeightOrigin;
	f32 ambientMediumHeightFalloff;
	u32 pad5;
	u32 pad6;
	u32 pad7;
	u64 materialMediumTableVa;
	u64 materialMediumRecordsVa;
	u32 materialMediumModelCount;
	u32 sceneEmitterLightCount;
	u64 sceneEmitterLightsVa;
	u32 meshNeeEnabled;
	u32 envNeeEnabled;
	u32 environmentMode;
	f32 environmentIntensity;
	f32 environmentRotation;
	u32 environmentRadianceHandle;
	u32 environmentPdfHandle;
	u32 environmentAliasProbHandle;
	u32 environmentAliasIndexHandle;
	u32 environmentEmitterIndex;
	f32 groundHeatingGroundHeight;
	f32 groundHeatingHazeHeight;
	f32 groundHeatingGradient;
	f32 groundHeatingNoiseAmplitude;
	f32 groundHeatingNoiseScale;
	u32 groundHeatingEnabled;
	u64 regirGuardsVa;
	u64 regirCellsVa;
	u64 regirReservoirsVa;
	u32 regirCellCount;
	f32 regirCellSize;
	u32 regirEnabled;
	u32 regirFillIndex;
	u32 regirMaxAge;
	u32 regirRefresh;
	u32 regirJitter;
	u32 regirPad0;
	u64 wavefrontTransportStatesVa;
	u64 wavefrontPathStatesVa;
	u64 wavefrontHitRecordsVa;
	u64 wavefrontNeeRecordsVa;
	u32 zeltnerLtcParamHandle;
	u32 passSlotFirst;
	u32 passSlotCount;
	u32 passPathCount;
	u32v2 passRegionOrigin;
	u32v2 passRegionExtent;
	u32 passRegionTilesW;
	f32 environmentBackgroundIntensity;
	f32 cameraLensRadius;
	f32 cameraFocalDistance;
};
static_assert(offsetof(GpuGlobal, rgb2specResolution) == 8u);
static_assert(offsetof(GpuGlobal, frameIndex) == 12u);
static_assert(offsetof(GpuGlobal, modelsVa) == 16u);
static_assert(offsetof(GpuGlobal, renderWidth) == 24u);
static_assert(offsetof(GpuGlobal, cameraOri) == 32u);
static_assert(offsetof(GpuGlobal, shutterOpen) == 44u);
static_assert(offsetof(GpuGlobal, cameraForward) == 48u);
static_assert(offsetof(GpuGlobal, shutterDuration) == 60u);
static_assert(offsetof(GpuGlobal, cameraRight) == 64u);
static_assert(offsetof(GpuGlobal, sampleIndex) == 76u);
static_assert(offsetof(GpuGlobal, cameraUp) == 80u);
static_assert(offsetof(GpuGlobal, blackholeMass) == 92u);
static_assert(offsetof(GpuGlobal, blackholeOri) == 96u);
static_assert(offsetof(GpuGlobal, blackholeLensRadius) == 108u);
static_assert(offsetof(GpuGlobal, blackholeMaxChords) == 112u);
static_assert(offsetof(GpuGlobal, blackholeDiskInnerRadius) == 116u);
static_assert(offsetof(GpuGlobal, blackholeDiskBrightness) == 128u);
static_assert(offsetof(GpuGlobal, blackholeDiskOpticalDepth) == 132u);
static_assert(offsetof(GpuGlobal, blackholeDiskOpticalDepthExponent) == 136u);
static_assert(offsetof(GpuGlobal, blackholeStepFactor) == 140u);
static_assert(offsetof(GpuGlobal, blackholeDiskTime) == 144u);
static_assert(offsetof(GpuGlobal, blackholeDiskHotspotSize) == 168u);
static_assert(offsetof(GpuGlobal, blackholeSpin) == 172u);
static_assert(offsetof(GpuGlobal, debugPathVa) == 176u);
static_assert(offsetof(GpuGlobal, debugPixel) == 184u);
static_assert(offsetof(GpuGlobal, debugCameraOri) == 192u);
static_assert(offsetof(GpuGlobal, debugCameraUp) == 240u);
static_assert(offsetof(GpuGlobal, ambientMediumSigmaAbsorption) == 256u);
static_assert(offsetof(GpuGlobal, ambientMediumSigmaScattering) == 272u);
static_assert(offsetof(GpuGlobal, ambientMediumAnisotropy) == 288u);
static_assert(offsetof(GpuGlobal, ambientMediumHeightFalloff) == 304u);
static_assert(offsetof(GpuGlobal, materialMediumTableVa) == 320u);
static_assert(offsetof(GpuGlobal, materialMediumModelCount) == 336u);
static_assert(offsetof(GpuGlobal, regirGuardsVa) == 416u);
static_assert(offsetof(GpuGlobal, regirCellCount) == 440u);
static_assert(offsetof(GpuGlobal, regirFillIndex) == 452u);
static_assert(offsetof(GpuGlobal, regirMaxAge) == 456u);
static_assert(offsetof(GpuGlobal, regirJitter) == 464u);
static_assert(offsetof(GpuGlobal, wavefrontTransportStatesVa) == 472u);
static_assert(offsetof(GpuGlobal, wavefrontPathStatesVa) == 480u);
static_assert(offsetof(GpuGlobal, wavefrontHitRecordsVa) == 488u);
static_assert(offsetof(GpuGlobal, wavefrontNeeRecordsVa) == 496u);
static_assert(offsetof(GpuGlobal, zeltnerLtcParamHandle) == 504u);
static_assert(offsetof(GpuGlobal, passSlotFirst) == 508u);
static_assert(offsetof(GpuGlobal, passRegionOrigin) == 520u);
static_assert(offsetof(GpuGlobal, passRegionTilesW) == 536u);
static_assert(offsetof(GpuGlobal, environmentBackgroundIntensity) == 540u);
static_assert(offsetof(GpuGlobal, cameraLensRadius) == 544u);
static_assert(offsetof(GpuGlobal, cameraFocalDistance) == 548u);
static_assert(sizeof(GpuGlobal) == 552u);

// mirrors WavefrontTransportState in wavefront/common.slang; only the size is
// read here
struct GpuWavefrontTransportState {
	f32v3 ori;
	f32v3 dir;
	f32 lambdas[4];
	f32 lambdaPdfs[4];
	f32 cameraTime;
	f32 pathLength;
	u32 mediumHandles[4];
	u32 mediumCount;
};
static_assert(sizeof(GpuWavefrontTransportState) == 84u);

// mirrors PathState in path/path-state.slang for the spectral build; the rgb
// build's is smaller (f32v3 spectra), so this sizes the buffer for both.
// bools are stored as 32-bit ints
struct GpuPathState {
	f32 throughput[4];
	f32 radiance[4];
	f32v3 scatterPosition;
	f32 scatterPdfBsdf;
	u32 scatterNeeAvailable;
	f32v3 scatterDirection;
	u32 scatterRegirSlot;
	i32 surfaceBounce;
	u32 alive;
};
static_assert(sizeof(GpuPathState) == 76u);

// ArrayStride of PathHitRecord (path/path-state.slang) in the spectral build,
// read from its spir-v; the rgb build's is 192
static constexpr u64 skPathHitRecordByteCount = 212u;
// ArrayStride of PathNeeRecord (path/sample-path.slang) in the spectral build;
// the rgb build's is 700
static constexpr u64 skPathNeeRecordByteCount = 736u;

// mirrors skPathBounceMax in path/path-state.slang
static constexpr u32 skWavefrontBounceMax = 8u;
static_assert(
	ponder::skWavefrontPassStepCount == 1u + 3u * skWavefrontBounceMax + 1u
);

// mirrors the camera entry's stage push constants (at offset 128)
struct WavefrontCameraPc {
	f32v2 globalJitter;
	u32 globalJitterEnabled;
};
static_assert(sizeof(WavefrontCameraPc) == 12u);

// mirrors the trace entry's stage push constants (at offset 128)
struct WavefrontTracePc {
	u32 bounce;
};
static_assert(sizeof(WavefrontTracePc) == 4u);

// mirrors the nee entry's stage push constants (at offset 128)
struct WavefrontNeePc {
	u32 bounce;
};
static_assert(sizeof(WavefrontNeePc) == 4u);

// mirrors the output entry's stage push constants (at offset 128)
struct WavefrontOutputPc {
	u32 outputHandle;
	f32 fireflyClampLuminance;
};
static_assert(sizeof(WavefrontOutputPc) == 8u);

// mirrors the integrate entry's stage push constants (at offset 128)
struct WavefrontIntegratePc {
	u32 outputHandle;
	u32 bounce;
	u32 aovEnabled;
	u32 aovPositionHandle;
	u32 aovNormalHandle;
	u32 aovAlbedoHandle;
	u32 rrGuideEnabled;
	u32 rrDiffuseAlbedoHandle;
	u32 rrSpecularAlbedoHandle;
	u32 rrRoughnessHandle;
	u32 rrSpecularHitDistanceHandle;
	u32 rrPositionHandle;
	u32 rrNormalHandle;
};
static_assert(sizeof(WavefrontIntegratePc) == 52u);

// mirrors the accumulate entry's stage push constants (at offset 128)
struct WavefrontAccumulatePc {
	u32 sampleHandle;
	u32 accumXyzHandle;
	u32 resolvedHandle;
	u32 outputHandle;
	u32 addSample;
	// the pass's first resetCount paths overwrite instead of accumulate
	u32 resetCount;
	f32 exposure;
};
static_assert(sizeof(WavefrontAccumulatePc) == 28u);

// path slots of a region: whole tiles, row-major
u32 region_tiles_w(ponder::WavefrontRegion const & region) {
	return (
		(region.extent.x + ponder::skWavefrontTileSize - 1u)
		/ ponder::skWavefrontTileSize
	);
}

u32 region_slot_count(ponder::WavefrontRegion const & region) {
	u32 const tilesH = (
		(region.extent.y + ponder::skWavefrontTileSize - 1u)
		/ ponder::skWavefrontTileSize
	);
	return region_tiles_w(region) * tilesH * ponder::skWavefrontTileSlotCount;
}

struct Rgb2SpecTable {
	u32 resolution;
	std::vector<f32> data;
};

Rgb2SpecTable rgb2spec_table_load(std::string const & path) {
	std::ifstream file(path, std::ios::binary);
	char header[4] {};
	u32 resolution = 0u;
	file.read(header, 4);
	file.read(reinterpret_cast<char *>(&resolution), sizeof(u32));
	if (!file || std::memcmp(header, "SPEC", 4) != 0 || resolution < 2u) {
		printf("[warn] wavefront: failed to load %s\n", path.c_str());
		return {};
	}
	u64 const resolution3 = (u64)resolution * resolution * resolution;
	u64 const floatCount = resolution + 9ull * resolution3;
	std::vector<f32> data(floatCount);
	file.read(
		reinterpret_cast<char *>(data.data()),
		(std::streamsize)(floatCount * sizeof(f32))
	);
	if (!file) {
		printf("[warn] wavefront: truncated %s\n", path.c_str());
		return {};
	}
	return Rgb2SpecTable { .resolution = resolution, .data = std::move(data) };
}

char const * const skIncludePaths[] = { skPonderShaderNewDir };
std::string const skShaderCacheDir = (
	std::string(skPonderShaderNewDir) + ".shader-cache"
);
std::string const skIntegratePath = (
	std::string(skPonderShaderNewDir) + "entry/wavefront-integrate.slang"
);
std::string const skAccumulatePath = (
	std::string(skPonderShaderNewDir) + "entry/wavefront-accumulate.slang"
);
std::string const skCameraPath = (
	std::string(skPonderShaderNewDir) + "entry/wavefront-camera.slang"
);
std::string const skTracePath = (
	std::string(skPonderShaderNewDir) + "entry/wavefront-trace.slang"
);
std::string const skNeePath = (
	std::string(skPonderShaderNewDir) + "entry/wavefront-nee.slang"
);
std::string const skOutputPath = (
	std::string(skPonderShaderNewDir) + "entry/wavefront-output.slang"
);
char const * const skRgbDefines[] = { "PONDER_SPECTRUM_RGB" };
char const * const skDebugDefines[] = { "PONDER_DEBUG_PATH" };
char const * const skRgbDebugDefines[] = {
	"PONDER_SPECTRUM_RGB",
	"PONDER_DEBUG_PATH",
};
// the trace kernel for paths that reach the black hole
char const * const skCurvedDefines[] = { "PONDER_TRACE_CURVED" };
char const * const skCurvedRgbDefines[] = {
	"PONDER_TRACE_CURVED",
	"PONDER_SPECTRUM_RGB",
};
char const * const skCurvedDebugDefines[] = {
	"PONDER_TRACE_CURVED",
	"PONDER_DEBUG_PATH",
};
char const * const skCurvedRgbDebugDefines[] = {
	"PONDER_TRACE_CURVED",
	"PONDER_SPECTRUM_RGB",
	"PONDER_DEBUG_PATH",
};

vkof::PipelineComputeCreateInfo compute_create_info(
	std::string const & path,
	srat::slice<char const * const> const defines = {},
	char const * const debugName = nullptr
) {
	return vkof::PipelineComputeCreateInfo {
		.pathCompute = path.c_str(),
		.defines = defines,
		.includePaths = srat::slice { skIncludePaths, 1u },
		.debugName = debugName,
	};
}

char const * const skTraceCurvedName = "wavefront-trace-curved";

vkof::Pipeline kernel_select(
	ponder::WavefrontKernel const & kernel,
	bool const spectral,
	bool const debugPath
) {
	if (debugPath) {
		return spectral ? kernel.spectralDebug : kernel.rgbDebug;
	}
	return spectral ? kernel.spectral : kernel.rgb;
}

void kernel_destroy(ponder::WavefrontKernel & kernel) {
	vkof::pipeline_destroy(kernel.spectral);
	vkof::pipeline_destroy(kernel.rgb);
	if (kernel.spectralDebug.id) {
		vkof::pipeline_destroy(kernel.spectralDebug);
		vkof::pipeline_destroy(kernel.rgbDebug);
	}
	kernel = {};
}

} // namespace

ponder::Wavefront ponder::wavefront_create() {
	// -- entry kernels come from shaderd when it has built them
	vkof::shader_cache_configure({
		.sourceRoot = skPonderShaderNewDir,
		.cacheDir = skShaderCacheDir.c_str(),
	});
	Rgb2SpecTable const table = rgb2spec_table_load(
		std::string(skPonderAssetDir) + "rgb2spec-srgb-64.coeff"
	);
	u64 const tableByteCount = table.data.size() * sizeof(f32);
	vkof::Buffer rgb2specBuffer {};
	if (tableByteCount > 0u) {
		rgb2specBuffer = (
			vkof::buffer_create({
				.byteCount = tableByteCount,
				.memory = vkof::BufferMemory::DeviceOnly,
			})
		);
		vkof::buffer_upload({
			.buffer = rgb2specBuffer,
			.byteOffset = 0u,
			.data = (
				srat::slice<u8 const>(
					reinterpret_cast<u8 const *>(table.data.data()),
					(u32)tableByteCount
				)
			),
		});
	}

	auto const globalBuffer = (
		vkof::buffer_create({
			.byteCount = sizeof(GpuGlobal),
			.memory = vkof::BufferMemory::DeviceOnly,
		})
	);

	auto const debugPathBuffer = (
		vkof::buffer_create({
			.byteCount = sizeof(DebugPath),
			.memory = vkof::BufferMemory::DeviceOnly,
		})
	);

	vkof::PipelineComputeCreateInfo const createInfos[] = {
		compute_create_info(skCameraPath),
		compute_create_info(skCameraPath, srat::slice { skRgbDefines, 1u }),
		compute_create_info(skTracePath),
		compute_create_info(skTracePath, srat::slice { skRgbDefines, 1u }),
		compute_create_info(
			skTracePath, srat::slice { skCurvedDefines, 1u }, skTraceCurvedName
		),
		compute_create_info(
			skTracePath, srat::slice { skCurvedRgbDefines, 2u }, skTraceCurvedName
		),
		compute_create_info(skIntegratePath),
		compute_create_info(skIntegratePath, srat::slice { skRgbDefines, 1u }),
		compute_create_info(skNeePath),
		compute_create_info(skNeePath, srat::slice { skRgbDefines, 1u }),
		compute_create_info(skOutputPath),
		compute_create_info(skOutputPath, srat::slice { skRgbDefines, 1u }),
		compute_create_info(skAccumulatePath),
	};
	vkof::Pipeline pipelines[13] {};
	vkof::pipeline_compute_create_batch(
		srat::slice { createInfos, 13u },
		srat::slice { pipelines, 13u }
	);
	auto const wavefront = Wavefront {
		.camera = WavefrontKernel {
			.spectral = pipelines[0],
			.rgb = pipelines[1],
		},
		.trace = WavefrontKernel {
			.spectral = pipelines[2],
			.rgb = pipelines[3],
		},
		.traceCurved = WavefrontKernel {
			.spectral = pipelines[4],
			.rgb = pipelines[5],
		},
		.integrate = WavefrontKernel {
			.spectral = pipelines[6],
			.rgb = pipelines[7],
		},
		.nee = WavefrontKernel {
			.spectral = pipelines[8],
			.rgb = pipelines[9],
		},
		.output = WavefrontKernel {
			.spectral = pipelines[10],
			.rgb = pipelines[11],
		},
		.accumulatePipeline = pipelines[12],
		.rgb2specBuffer = rgb2specBuffer,
		.rgb2specVa = (
			rgb2specBuffer.id ? vkof::buffer_virtual_address(rgb2specBuffer) : 0u
		),
		.rgb2specResolution = table.resolution,
		.globalBuffer = globalBuffer,
		.globalVa = vkof::buffer_virtual_address(globalBuffer),
		.debugPathBuffer = debugPathBuffer,
		.debugPathVa = vkof::buffer_virtual_address(debugPathBuffer),
		.zeltnerTables = ponder::zeltner_tables_create(),
	};
	return wavefront;
}

void ponder::wavefront_destroy(Wavefront & wavefront) {
	kernel_destroy(wavefront.camera);
	kernel_destroy(wavefront.trace);
	kernel_destroy(wavefront.traceCurved);
	kernel_destroy(wavefront.integrate);
	kernel_destroy(wavefront.nee);
	kernel_destroy(wavefront.output);
	vkof::pipeline_destroy(wavefront.accumulatePipeline);
	vkof::buffer_destroy(wavefront.debugPathBuffer);
	if (wavefront.rgb2specBuffer.id) {
		vkof::buffer_destroy(wavefront.rgb2specBuffer);
	}
	vkof::buffer_destroy(wavefront.globalBuffer);
	ponder::zeltner_tables_destroy(wavefront.zeltnerTables);
	if (wavefront.accumXyzImage.id) {
		vkof::image_destroy(wavefront.accumXyzImage);
	}
	if (wavefront.transportStateBuffer.id) {
		vkof::buffer_destroy(wavefront.transportStateBuffer);
		vkof::buffer_destroy(wavefront.pathStateBuffer);
		vkof::buffer_destroy(wavefront.hitRecordBuffer);
		vkof::buffer_destroy(wavefront.neeRecordBuffer);
	}
	wavefront = {};
}

void ponder::wavefront_resize(
	Wavefront & wavefront,
	u32 const width,
	u32 const height,
	u32 const pathBudget
) {
	wavefront.passStep = 0u;
	wavefront.slotCursor = 0u;
	wavefront.resetRemaining = 0u;
	// whole tiles, no more than the image's
	u32 const pathCapacity = (
		std::min(
			std::max(
				pathBudget / skWavefrontTileSlotCount * skWavefrontTileSlotCount,
				skWavefrontTileSlotCount
			),
			region_slot_count(WavefrontRegion {
				.origin = { 0u, 0u },
				.extent = { width, height },
			})
		)
	);
	wavefront.pathCapacity = pathCapacity;
	if (wavefront.accumXyzImage.id) {
		vkof::image_destroy(wavefront.accumXyzImage);
	}
	wavefront.accumXyzImage = (
		vkof::image_create({
			.width = width,
			.height = height,
			.depth = 1u,
			.format = vkof::ImageFormat::r32g32b32a32_sfloat,
			.mipLevels = 1u,
			.optInitialData = {},
		})
	);
	wavefront.accumXyzHandle = (
		vkof::image_storage_handle({
			.image = wavefront.accumXyzImage,
			.mipLevel = 0u,
		})
	);
	if (wavefront.transportStateBuffer.id) {
		vkof::buffer_destroy(wavefront.transportStateBuffer);
		vkof::buffer_destroy(wavefront.pathStateBuffer);
		vkof::buffer_destroy(wavefront.hitRecordBuffer);
		vkof::buffer_destroy(wavefront.neeRecordBuffer);
	}
	wavefront.transportStateBuffer = (
		vkof::buffer_create({
			.byteCount = (
				(u64)sizeof(GpuWavefrontTransportState) * pathCapacity
			),
			.memory = vkof::BufferMemory::DeviceOnly,
		})
	);
	wavefront.transportStateVa = (
		vkof::buffer_virtual_address(wavefront.transportStateBuffer)
	);
	wavefront.pathStateBuffer = (
		vkof::buffer_create({
			.byteCount = (u64)sizeof(GpuPathState) * pathCapacity,
			.memory = vkof::BufferMemory::DeviceOnly,
		})
	);
	wavefront.pathStateVa = (
		vkof::buffer_virtual_address(wavefront.pathStateBuffer)
	);
	wavefront.hitRecordBuffer = (
		vkof::buffer_create({
			.byteCount = skPathHitRecordByteCount * pathCapacity,
			.memory = vkof::BufferMemory::DeviceOnly,
		})
	);
	wavefront.hitRecordVa = (
		vkof::buffer_virtual_address(wavefront.hitRecordBuffer)
	);
	wavefront.neeRecordBuffer = (
		vkof::buffer_create({
			.byteCount = skPathNeeRecordByteCount * pathCapacity,
			.memory = vkof::BufferMemory::DeviceOnly,
		})
	);
	wavefront.neeRecordVa = (
		vkof::buffer_virtual_address(wavefront.neeRecordBuffer)
	);
}

ponder::WavefrontCameraBasis ponder::wavefront_camera_basis(
	WavefrontCamera const & camera,
	f32 const aspect
) {
	f32v3 const forward = f32v3_normalize(camera.forward);
	f32v3 const right = f32v3_normalize(f32v3_cross(forward, skWorldUp));
	f32v3 const up = f32v3_cross(right, forward);
	f32 const tanHalfFovY = std::tan(camera.fovY * 0.5f);
	return WavefrontCameraBasis {
		.forward = forward,
		.right = right * (tanHalfFovY * aspect),
		.up = up * tanHalfFovY,
	};
}

void ponder::wavefront_global_update(
	Wavefront & wavefront,
	Regir const & regir,
	WavefrontGlobalInfo const & info
) {
	// -- a new pass takes the current indices
	if (wavefront.passStep == 0u) {
		wavefront.passSampleIndex = info.sampleIndex;
		wavefront.passRegirFillIndex = info.regir.fillIndex;
	}
	// -- a new pass takes the region's next paths
	if (wavefront.passStep == 0u) {
		auto const full = WavefrontRegion {
			.origin = { 0u, 0u },
			.extent = { info.width, info.height },
		};
		auto const fnClamped = [&](WavefrontRegion const & r) {
			u32v2 const origin = {
				std::min(r.origin.x, info.width - 1u),
				std::min(r.origin.y, info.height - 1u),
			};
			return WavefrontRegion {
				.origin = origin,
				.extent = {
					std::clamp(r.extent.x, 1u, info.width - origin.x),
					std::clamp(r.extent.y, 1u, info.height - origin.y),
				},
			};
		};
		auto const fnContains = [](WavefrontRegion const & r, u32v2 const p) {
			return (
				p.x >= r.origin.x && p.x - r.origin.x < r.extent.x
				&& p.y >= r.origin.y && p.y - r.origin.y < r.extent.y
			);
		};
		WavefrontRegion region = fnClamped(info.region.value_or(full));
		// a debug capture traces its pixel
		if (info.debugPath && !fnContains(region, info.debugPixel)) {
			region = full;
		}
		u32 const slotCount = region_slot_count(region);
		if (wavefront.slotCursor >= slotCount) {
			wavefront.slotCursor = 0u;
		}
		u32 slotFirst = wavefront.slotCursor;
		// a partial debug pass starts at the debug pixel's tile instead, and
		// leaves the cursor
		wavefront.passHoldsCursor = (
			info.debugPath && wavefront.pathCapacity < slotCount
		);
		if (wavefront.passHoldsCursor) {
			u32v2 const tile = {
				(info.debugPixel.x - region.origin.x) / skWavefrontTileSize,
				(info.debugPixel.y - region.origin.y) / skWavefrontTileSize,
			};
			slotFirst = (
				(tile.y * region_tiles_w(region) + tile.x)
				* skWavefrontTileSlotCount
			);
		}
		wavefront.passRegion = region;
		wavefront.passSlotCount = slotCount;
		wavefront.passSlotFirst = slotFirst;
		// a pass ends at its sweep's end, so a sweep is exactly one sample
		wavefront.passPathCount = (
			std::min(wavefront.pathCapacity, slotCount - slotFirst)
		);
	}
	f32 const aspect = (f32)info.width / (f32)info.height;
	WavefrontCameraBasis const view = wavefront_camera_basis(info.camera, aspect);
	WavefrontCameraBasis const debugView = (
		wavefront_camera_basis(info.debugCamera, aspect)
	);
	auto const global = GpuGlobal {
		.rgb2specVa = wavefront.rgb2specVa,
		.rgb2specResolution = wavefront.rgb2specResolution,
		.frameIndex = info.frameIndex,
		.modelsVa = info.modelsVa,
		.renderWidth = info.width,
		.renderHeight = info.height,
		.cameraOri = info.camera.position,
		.shutterOpen = info.shutterOpen,
		.cameraForward = view.forward,
		.shutterDuration = info.shutterDuration,
		.cameraRight = view.right,
		.sampleIndex = wavefront.passSampleIndex,
		.cameraUp = view.up,
		.blackholeMass = info.blackhole.mass,
		.blackholeOri = info.blackhole.ori,
		.blackholeLensRadius = info.blackhole.lensRadius,
		.blackholeMaxChords = info.blackhole.maxChords,
		.blackholeDiskInnerRadius = info.blackhole.diskInnerRadius,
		.blackholeDiskOuterRadius = info.blackhole.diskOuterRadius,
		.blackholeDiskTemperature = info.blackhole.diskTemperature,
		.blackholeDiskBrightness = info.blackhole.diskBrightness,
		.blackholeDiskOpticalDepth = info.blackhole.diskOpticalDepth,
		.blackholeDiskOpticalDepthExponent = info.blackhole.diskOpticalDepthExponent,
		.blackholeStepFactor = info.blackhole.stepFactor,
		.blackholeDiskTime = info.blackhole.diskTime,
		.blackholeDiskRotation = info.blackhole.diskRotation,
		.blackholeDiskNoiseAmplitude = info.blackhole.diskNoiseAmplitude,
		.blackholeDiskNoiseScale = info.blackhole.diskNoiseScale,
		.blackholeDiskHotspotRadius = info.blackhole.diskHotspotRadius,
		.blackholeDiskHotspotTemperature = info.blackhole.diskHotspotTemperature,
		.blackholeDiskHotspotSize = info.blackhole.diskHotspotSize,
		.blackholeSpin = info.blackhole.spin,
		.debugPathVa = info.debugPath ? wavefront.debugPathVa : 0u,
		.debugPixel = info.debugPixel,
		.debugCameraOri = info.debugCamera.position,
		.pad1 = 0u,
		.debugCameraForward = debugView.forward,
		.pad2 = 0u,
		.debugCameraRight = debugView.right,
		.pad3 = 0u,
		.debugCameraUp = debugView.up,
		.pad4 = 0u,
		.ambientMediumSigmaAbsorption = info.ambientMedium.sigmaAbsorption,
		.ambientMediumType = (u32)info.ambientMedium.type,
		.ambientMediumSigmaScattering = info.ambientMedium.sigmaScattering,
		.ambientMediumPhase = (u32)info.ambientMedium.phase,
		.ambientMediumAnisotropy = info.ambientMedium.anisotropy,
		.ambientMediumDraineAlpha = info.ambientMedium.draineAlpha,
		.ambientMediumDropletDiameter = info.ambientMedium.dropletDiameter,
		.ambientMediumHeightOrigin = info.ambientMedium.heightOrigin,
		.ambientMediumHeightFalloff = info.ambientMedium.heightFalloff,
		.pad5 = 0u,
		.pad6 = 0u,
		.pad7 = 0u,
		.materialMediumTableVa = info.materialMediumTableVa,
		.materialMediumRecordsVa = info.materialMediumRecordsVa,
		.materialMediumModelCount = info.materialMediumModelCount,
		.sceneEmitterLightCount = info.sceneEmitterLightCount,
		.sceneEmitterLightsVa = info.sceneEmitterLightsVa,
		.meshNeeEnabled = info.meshNeeEnabled ? 1u : 0u,
		.envNeeEnabled = info.envNeeEnabled ? 1u : 0u,
		.environmentMode = (u32)info.environment.mode,
		.environmentIntensity = info.environment.intensity,
		.environmentRotation = info.environment.rotation,
		.environmentRadianceHandle = info.environment.radianceHandle,
		.environmentPdfHandle = info.environment.pdfHandle,
		.environmentAliasProbHandle = info.environment.aliasProbHandle,
		.environmentAliasIndexHandle = info.environment.aliasIndexHandle,
		.environmentEmitterIndex = info.environment.emitterIndex,
		.groundHeatingGroundHeight = info.groundHeating.groundHeight,
		.groundHeatingHazeHeight = info.groundHeating.hazeHeight,
		.groundHeatingGradient = info.groundHeating.gradient,
		.groundHeatingNoiseAmplitude = info.groundHeating.noiseAmplitude,
		.groundHeatingNoiseScale = info.groundHeating.noiseScale,
		.groundHeatingEnabled = info.groundHeating.enabled ? 1u : 0u,
		.regirGuardsVa = vkof::buffer_virtual_address(regir.guardBuffer),
		.regirCellsVa = vkof::buffer_virtual_address(regir.cellBuffer),
		.regirReservoirsVa = vkof::buffer_virtual_address(regir.reservoirBuffer),
		.regirCellCount = regir.cellCount,
		.regirCellSize = info.regir.cellSize,
		.regirEnabled = info.regir.enabled ? 1u : 0u,
		.regirFillIndex = wavefront.passRegirFillIndex,
		.regirMaxAge = info.regir.maxAge,
		.regirRefresh = info.regir.refresh ? 1u : 0u,
		.regirJitter = info.regir.jitter ? 1u : 0u,
		.regirPad0 = 0u,
		.wavefrontTransportStatesVa = wavefront.transportStateVa,
		.wavefrontPathStatesVa = wavefront.pathStateVa,
		.wavefrontHitRecordsVa = wavefront.hitRecordVa,
		.wavefrontNeeRecordsVa = wavefront.neeRecordVa,
		.zeltnerLtcParamHandle = wavefront.zeltnerTables.zeltnerLtcParamHandle,
		.passSlotFirst = wavefront.passSlotFirst,
		.passSlotCount = wavefront.passSlotCount,
		.passPathCount = wavefront.passPathCount,
		.passRegionOrigin = wavefront.passRegion.origin,
		.passRegionExtent = wavefront.passRegion.extent,
		.passRegionTilesW = region_tiles_w(wavefront.passRegion),
		.environmentBackgroundIntensity = info.environment.backgroundIntensity,
		.cameraLensRadius = info.lensRadius,
		.cameraFocalDistance = info.focalDistance,
	};
	vkof::buffer_upload({
		.buffer = wavefront.globalBuffer,
		.byteOffset = 0u,
		.data = srat::slice_as_bytes(global),
	});
}

void ponder::wavefront_pass_restart(Wavefront & wavefront) {
	wavefront.passStep = 0u;
}

ponder::WavefrontIntegrateResult ponder::wavefront_integrate(
	Wavefront & wavefront,
	WavefrontIntegrateInfo const & info
) {
	// -- rgb and spectral paths have different record layouts
	if (wavefront.passStep != 0u && wavefront.passSpectral != info.spectral) {
		wavefront.passStep = 0u;
	}
	if (wavefront.passStep == 0u) {
		wavefront.passSpectral = info.spectral;
		// -- the pass takes its paths from the cursor
		u32 const slotLast = wavefront.passSlotFirst + wavefront.passPathCount;
		bool const sweepEnds = slotLast == wavefront.passSlotCount;
		wavefront.passSweepCount = 0u;
		if (!wavefront.passHoldsCursor) {
			wavefront.passSweepCount = sweepEnds ? 1u : 0u;
			wavefront.slotCursor = sweepEnds ? 0u : slotLast;
		}
	}

	// -- the steps this node dispatches
	u32 const stepFirst = wavefront.passStep;
	u32 const stepLast = (
		std::min(
			stepFirst + std::max(info.stepBudget, 1u),
			skWavefrontPassStepCount
		)
	);
	bool const passComplete = stepLast == skWavefrontPassStepCount;
	wavefront.passStep = passComplete ? 0u : stepLast;

	auto const node = (
		vkof::render_node_create({
			.queue = vkof::CommandQueue::graphics,
			.debugName = "wavefront",
		})
	);
	vkof::render_node_add_image({
		.node = node,
		.image = info.outputImage,
		.access = vkof::RenderNodeAccess::write,
	});
	// -- aovs, written by the first shade step
	bool const aovStep = (
		info.aov.has_value() && stepFirst <= 2u && stepLast > 2u
	);
	WavefrontAov const aov = info.aov.value_or(WavefrontAov {});
	if (aovStep) {
		for (vkof::Image const image : {
			aov.positionImage, aov.normalImage, aov.albedoImage
		}) {
			vkof::render_node_add_persistent_image({
				.node = node,
				.image = image,
				.access = vkof::RenderNodeAccess::write,
			});
		}
	}
	bool const rrGuideStep = aovStep && info.rrGuide.has_value();
	WavefrontRrGuide const rrGuide = info.rrGuide.value_or(WavefrontRrGuide {});
	if (rrGuideStep) {
		for (vkof::Image const image : {
			rrGuide.diffuseAlbedoImage,
			rrGuide.specularAlbedoImage,
			rrGuide.roughnessImage,
			rrGuide.specularHitDistanceImage,
			rrGuide.positionImage,
			rrGuide.normalImage,
		}) {
			vkof::render_node_add_persistent_image({
				.node = node,
				.image = image,
				.access = vkof::RenderNodeAccess::write,
			});
		}
	}
	auto const cameraPc = WavefrontCameraPc {
		.globalJitter = info.globalJitter.value_or(f32v2 { 0.0f, 0.0f }),
		.globalJitterEnabled = info.globalJitter.has_value() ? 1u : 0u,
	};
	vkof::Pipeline const cameraPipeline = (
		kernel_select(wavefront.camera, info.spectral, info.debugPath)
	);
	vkof::Pipeline const tracePipeline = (
		kernel_select(wavefront.trace, info.spectral, info.debugPath)
	);
	vkof::Pipeline const traceCurvedPipeline = (
		kernel_select(wavefront.traceCurved, info.spectral, info.debugPath)
	);
	vkof::Pipeline const integratePipeline = (
		kernel_select(wavefront.integrate, info.spectral, info.debugPath)
	);
	vkof::Pipeline const neePipeline = (
		kernel_select(wavefront.nee, info.spectral, info.debugPath)
	);
	vkof::Pipeline const outputPipeline = (
		kernel_select(wavefront.output, info.spectral, info.debugPath)
	);
	u32 const outputHandle = info.outputHandle;
	f32 const fireflyClampLuminance = info.fireflyClampLuminance;
	u32 const pathCount = wavefront.passPathCount;
	vkof::render_node_callback({
		.node = node,
		.callback = [=](vkof::CommandBuffer const & cmd) {
			auto const fnDispatch = (
				[&](vkof::Pipeline const pipeline, auto const push) {
					vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
						.cmd = cmd,
						.pipeline = pipeline,
						.push = push,
						.threadgroupSize = (
							u32v3 { skWavefrontTileSlotCount, 1u, 1u }
						),
						.invocationCount = u32v3 { pathCount, 1u, 1u },
					});
				}
			);
			for (u32 step = stepFirst; step < stepLast; ++step) {
				// the path buffers are only touched by this node, so the
				// barriers order every access to them, across frames too
				vkof::cmd_barrier_compute(cmd);
				// -- camera
				if (step == 0u) {
					fnDispatch(cameraPipeline, cameraPc);
					continue;
				}
				// -- output
				if (step == skWavefrontPassStepCount - 1u) {
					fnDispatch(
						outputPipeline,
						WavefrontOutputPc {
							.outputHandle = outputHandle,
							.fireflyClampLuminance = fireflyClampLuminance,
						}
					);
					continue;
				}
				// -- one stage of a bounce
				u32 const bounce = (step - 1u) / 3u;
				switch ((step - 1u) % 3u) {
					case 0u: {
						// each path takes exactly one of the two traces
						auto const pc = WavefrontTracePc { .bounce = bounce };
						fnDispatch(tracePipeline, pc);
						fnDispatch(traceCurvedPipeline, pc);
					} break;
					case 1u: {
						fnDispatch(
							integratePipeline,
							WavefrontIntegratePc {
								.outputHandle = outputHandle,
								.bounce = bounce,
								.aovEnabled = aovStep ? 1u : 0u,
								.aovPositionHandle = aov.positionHandle,
								.aovNormalHandle = aov.normalHandle,
								.aovAlbedoHandle = aov.albedoHandle,
								.rrGuideEnabled = rrGuideStep ? 1u : 0u,
								.rrDiffuseAlbedoHandle = rrGuide.diffuseAlbedoHandle,
								.rrSpecularAlbedoHandle = rrGuide.specularAlbedoHandle,
								.rrRoughnessHandle = rrGuide.roughnessHandle,
								.rrSpecularHitDistanceHandle = (
									rrGuide.specularHitDistanceHandle
								),
								.rrPositionHandle = rrGuide.positionHandle,
								.rrNormalHandle = rrGuide.normalHandle,
							}
						);
					} break;
					default: {
						fnDispatch(neePipeline, WavefrontNeePc { .bounce = bounce });
					} break;
				}
			}
		},
	});
	return WavefrontIntegrateResult {
		.node = node,
		.stepCount = stepLast - stepFirst,
		.passComplete = passComplete,
		.sweepCount = passComplete ? wavefront.passSweepCount : 0u,
	};
}

void ponder::wavefront_debug_path_prepare(Wavefront & wavefront) {
	if (wavefront.integrate.spectralDebug.id) {
		return;
	}
	vkof::PipelineComputeCreateInfo createInfos[] = {
		compute_create_info(skCameraPath, srat::slice { skDebugDefines, 1u }),
		compute_create_info(skCameraPath, srat::slice { skRgbDebugDefines, 2u }),
		compute_create_info(skTracePath, srat::slice { skDebugDefines, 1u }),
		compute_create_info(skTracePath, srat::slice { skRgbDebugDefines, 2u }),
		compute_create_info(
			skTracePath,
			srat::slice { skCurvedDebugDefines, 2u },
			skTraceCurvedName
		),
		compute_create_info(
			skTracePath,
			srat::slice { skCurvedRgbDebugDefines, 3u },
			skTraceCurvedName
		),
		compute_create_info(skIntegratePath, srat::slice { skDebugDefines, 1u }),
		compute_create_info(skIntegratePath, srat::slice { skRgbDebugDefines, 2u }),
		compute_create_info(skNeePath, srat::slice { skDebugDefines, 1u }),
		compute_create_info(skNeePath, srat::slice { skRgbDebugDefines, 2u }),
		compute_create_info(skOutputPath, srat::slice { skDebugDefines, 1u }),
		compute_create_info(skOutputPath, srat::slice { skRgbDebugDefines, 2u }),
	};
	// shaderd builds these after the kernels every frame needs
	for (vkof::PipelineComputeCreateInfo & createInfo : createInfos) {
		createInfo.deferred = true;
	}
	vkof::Pipeline pipelines[12] {};
	vkof::pipeline_compute_create_batch(
		srat::slice { createInfos, 12u },
		srat::slice { pipelines, 12u }
	);
	wavefront.camera.spectralDebug = pipelines[0];
	wavefront.camera.rgbDebug = pipelines[1];
	wavefront.trace.spectralDebug = pipelines[2];
	wavefront.trace.rgbDebug = pipelines[3];
	wavefront.traceCurved.spectralDebug = pipelines[4];
	wavefront.traceCurved.rgbDebug = pipelines[5];
	wavefront.integrate.spectralDebug = pipelines[6];
	wavefront.integrate.rgbDebug = pipelines[7];
	wavefront.nee.spectralDebug = pipelines[8];
	wavefront.nee.rgbDebug = pipelines[9];
	wavefront.output.spectralDebug = pipelines[10];
	wavefront.output.rgbDebug = pipelines[11];
}

void ponder::wavefront_debug_path_read(
	Wavefront const & wavefront,
	DebugPath & path
) {
	vkof::buffer_download({
		.buffer = wavefront.debugPathBuffer,
		.byteOffset = 0u,
		.dst = srat::slice<u8>(reinterpret_cast<u8 *>(&path), (u32)sizeof(DebugPath)),
	});
}

vkof::RenderNode ponder::wavefront_accumulate(
	Wavefront & wavefront,
	WavefrontAccumulateInfo const & info
) {
	// -- a reset covers the next sweep's paths, pass by pass
	u32 resetCount = 0u;
	if (info.addSample) {
		if (info.reset) {
			wavefront.resetRemaining = wavefront.passSlotCount;
		}
		resetCount = std::min(wavefront.resetRemaining, wavefront.passPathCount);
		wavefront.resetRemaining -= resetCount;
	}
	auto const node = (
		vkof::render_node_create({
			.queue = vkof::CommandQueue::graphics,
			.debugName = "accumulate",
		})
	);
	if (info.addSample) {
		vkof::render_node_add_image({
			.node = node,
			.image = info.sampleImage,
			.access = vkof::RenderNodeAccess::read,
		});
	}
	vkof::render_node_add_persistent_image({
		.node = node,
		.image = wavefront.accumXyzImage,
		.access = vkof::RenderNodeAccess::readWrite,
	});
	vkof::render_node_add_persistent_image({
		.node = node,
		.image = info.resolvedImage,
		.access = vkof::RenderNodeAccess::write,
	});
	vkof::render_node_add_image({
		.node = node,
		.image = info.outputImage,
		.access = vkof::RenderNodeAccess::write,
	});
	vkof::Pipeline const pipeline = wavefront.accumulatePipeline;
	auto const pc = WavefrontAccumulatePc {
		.sampleHandle = info.sampleHandle,
		.accumXyzHandle = wavefront.accumXyzHandle,
		.resolvedHandle = info.resolvedHandle,
		.outputHandle = info.outputHandle,
		.addSample = info.addSample ? 1u : 0u,
		.resetCount = resetCount,
		.exposure = info.exposure,
	};
	u32 const width = info.width;
	u32 const height = info.height;
	vkof::render_node_callback({
		.node = node,
		.callback = [=](vkof::CommandBuffer const & cmd) {
			vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
				.cmd = cmd,
				.pipeline = pipeline,
				.push = pc,
				.threadgroupSize = u32v3 { 16u, 16u, 1u },
				.invocationCount = u32v3 { width, height, 1u },
			});
		},
	});
	return node;
}
