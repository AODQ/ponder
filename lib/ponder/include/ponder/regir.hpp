#pragma once

#include <vkof/vkof.hpp>

namespace ponder {

// -- regir grid layouts, mirroring types/regir.slang (scalar layout)

// mirror the shader's skRegir* constants; renamed because the old pipeline's
// util-regir-shared.h defines those names as macros
constexpr u32 skWavefrontRegirReservoirCount = 64u;
constexpr u32 skWavefrontRegirNormalBucketNone = 16u;
// also the never-filled lastFillIndex
constexpr u32 skWavefrontRegirSlotInvalid = 0xFFFFFFFFu;
constexpr u32 skWavefrontRegirSlotEvicted = 0xFFFFFFFEu;

struct RegirCellKey {
	i32v3 gridCoord;
	u32 normalBucket;
};
static_assert(sizeof(RegirCellKey) == 16u);

struct RegirReservoir {
	u32 lightIndex;
	u32 primitiveIndex;
	f32v2 parameters;
	f32 weight;
	f32 targetValue;
};
static_assert(sizeof(RegirReservoir) == 24u);

struct RegirCell {
	RegirCellKey key;
	f32v3 representativeOri;
	f32v3 representativeNor;
	f32 normalizationNoncanonical;
	f32 normalizationCanonical;
	u32 lastFillIndex;
	u32 frameIndexLastUsed;
	f32v3 pendingOri;
	f32v3 pendingNor;
};
static_assert(sizeof(RegirCell) == 80u);

// world-space light-selection grid, gpu-written only
struct Regir {
	vkof::Pipeline fillPipeline {};
	vkof::Pipeline viewPipeline {};

	vkof::Buffer guardBuffer {};
	vkof::Buffer cellBuffer {};
	vkof::Buffer reservoirBuffer {};
	u32 cellCount {};
};

[[nodiscard]] Regir regir_create();
void regir_destroy(Regir & regir);

// per-frame settings, uploaded through the wavefront global
struct RegirSettings {
	bool enabled;
	f32 cellSize;
	u32 fillIndex;
	// frames to evict regir cell, 0 never evicts
	u32 maxAge;
	// lets shading points replace cell representatives; off while the camera
	// moves or the timeline is paused, where it only adds flicker
	bool refresh;
	// nee looks up a random neighbouring cell, so adjacent pixels don't share
	// reservoirs and cell seams blur out
	bool jitter;
};

// refills claimed cells; schedule before integrate
[[nodiscard]] vkof::RenderNode regir_fill(Regir const & regir);

// empties every regir slot; blocking
void regir_clear(Regir const & regir);

void regir_view_prepare(Regir & regir);

struct RegirViewInfo {
	vkof::TransientImage outputImage;
	u32 outputHandle;
	// 1 probe state, 2 a color per usable probe
	u32 mode;
	f32 exposure;
	u32 width;
	u32 height;
};

// replaces integrate, shading each pixel's primary hit by its regir probe
[[nodiscard]] vkof::RenderNode regir_view(
	Regir const & regir,
	RegirViewInfo const & info
);

} // namespace ponder
