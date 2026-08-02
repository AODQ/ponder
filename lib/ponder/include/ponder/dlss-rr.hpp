#pragma once

// DLSS ray reconstruction realtime mode: each frame is one pass, denoised by
// RR instead of accumulated. a stub that reports itself unsupported without
// PONDER_DLSS

#include <ponder/wavefront.hpp>

#include <vkof/vkof.hpp>

namespace ponder {

// mirrors vkof::NgxRrPreset
enum struct DlssRrPreset : u32 {
	Default = 0u,
	D = 4u,
	E = 5u,
	F = 6u,
};

// mirrors vkof::NgxRrQuality; below dlaa, rr upscales to the output size
enum struct DlssRrQuality : u32 {
	Performance = 0u,
	Balanced = 1u,
	Quality = 2u,
	UltraPerformance = 3u,
	Dlaa = 5u,
};

struct DlssRr {
	// created on first use
	bool created {};
	// the guides and inputs are renderSize, the output outputSize
	u32v2 renderSize {};
	u32v2 outputSize {};
	vkof::Pipeline preparePipeline {};
	vkof::Pipeline resolvePipeline {};
	// -- render-resolution guides, inputs and output
	vkof::Image colorImage {};
	u32 colorHandle {};
	vkof::Image diffuseAlbedoImage {};
	u32 diffuseAlbedoHandle {};
	vkof::Image specularAlbedoImage {};
	u32 specularAlbedoHandle {};
	vkof::Image roughnessImage {};
	u32 roughnessHandle {};
	vkof::Image depthImage {};
	u32 depthHandle {};
	vkof::Image motionImage {};
	u32 motionHandle {};
	vkof::Image specularHitDistanceImage {};
	u32 specularHitDistanceHandle {};
	// the guided surface, which smooth glass can see through
	vkof::Image positionImage {};
	u32 positionHandle {};
	vkof::Image normalImage {};
	u32 normalHandle {};
	vkof::Image outputImage {};
	u32 outputHandle {};
	// -- temporal state
	u32 jitterPhase {};
	bool hasPrevious {};
	WavefrontCamera previousCamera {};
	bool evaluateFailed {};
};

// false when the build lacks DLSS or the driver lacks ray reconstruction
[[nodiscard]] bool dlss_rr_supported();
[[nodiscard]] char const * dlss_rr_unsupported_reason();

// the size to trace at for an output size
[[nodiscard]] u32v2 dlss_rr_render_size(
	DlssRrQuality const quality,
	u32v2 const outputSize
);

// compiles the pipelines, and creates the images when absent or resized
void dlss_rr_prepare(
	DlssRr & rr,
	u32v2 const renderSize,
	u32v2 const outputSize
);
void dlss_rr_destroy(DlssRr & rr);

// the next frame's pixel offset in [0, 1), for WavefrontIntegrateInfo
[[nodiscard]] f32v2 dlss_rr_jitter(DlssRr const & rr);
[[nodiscard]] WavefrontRrGuide dlss_rr_guide(DlssRr const & rr);

struct DlssRrFrameInfo {
	// the finished pass, xyz
	vkof::TransientImage colorImage;
	u32 colorHandle;
	// the pass's primary hit aovs
	WavefrontAov aov;
	// written like wavefront_accumulate's resolved and output images
	vkof::Image resolvedImage;
	u32 resolvedHandle;
	vkof::TransientImage outputImage;
	u32 outputHandle;
	f32 exposure;
	// the camera the pass was traced with
	WavefrontCamera camera;
	DlssRrQuality quality;
	DlssRrPreset preset;
	// drops RR's history, for changes other than camera motion
	bool reset;
};

struct DlssRrNodes {
	vkof::RenderNode prepare;
	vkof::RenderNode evaluate;
	vkof::RenderNode resolve;
};

// prepare, evaluate and resolve, in order; advances the jitter and remembers
// the camera for the next frame's motion vectors
[[nodiscard]] DlssRrNodes dlss_rr_nodes(
	DlssRr & rr,
	DlssRrFrameInfo const & info
);

} // namespace ponder
