#pragma once

#include <ponder/debug-path.hpp>
#include <ponder/regir.hpp>

#include <vector>

namespace ponder {

// mirrors RegirDebugSample in entry/regir-debug.slang
struct RegirDebugSample {
	f32v3 position;
	f32 weight;
	f32v3 normal;
	f32 targetValue;
	u32 lightIndex;
	// 1 when the reservoir holds a usable sample
	u32 valid;
	// p̂ shading would report for this point, area measure
	f32 pdfArea;
	u32 pad0;
};
static_assert(sizeof(RegirDebugSample) == 48u);

// a claimed slot as read back from the grid
struct RegirDebugCell {
	u32 slot;
	// distance from the key's home slot along the linear probe
	u32 probeLength;
	RegirCell cell;
};

// grid statistics and the probe (cell) inspector of the viewer
struct RegirDebugState {
	// frames between grid readbacks; each one stalls on a ~4 MiB download
	i32 refreshInterval { 30 };
	bool autoRefresh { true };
	bool refreshRequested { true };
	u32 framesSinceRefresh { 0u };
	// set by regir_debug_node, consumed by regir_debug_readback
	bool refreshDue { false };
	bool samplesScheduled { false };

	// -- last readback
	bool valid { false };
	u32 fillIndex { 0u };
	f32 cellSize { 0.0f };
	u32 cellCount { 0u };
	u32 claimedCount { 0u };
	// filled by this frame's fill pass, the only ones shading reads
	u32 filledCount { 0u };
	// claimed during the last integrate, filled next frame
	u32 unfilledCount { 0u };
	// claimed but not refilled this frame; should stay 0 while regir is on
	u32 staleCount { 0u };
	// filled, yet no candidate had a nonzero target
	u32 zeroNormCount { 0u };
	u32 maxProbe { 0u };
	f32 meanProbe { 0.0f };
	u32 evictedCount { 0u };
	// frames since a lookup last found the cell
	u32 maxAge { 0u };
	f32 meanAge { 0.0f };
	std::vector<f32> probeHistogram {};
	std::vector<RegirDebugCell> cells {};

	// -- selected probe
	u32 selectedSlot { skWavefrontRegirSlotInvalid };
	// -1 for none
	i32 highlightedReservoir { -1 };
	bool drawSelected { true };
	// every claimed probe as a sphere at its representative, colored by state
	bool drawAllProbes { false };
	f32 drawAllMaxDistance { 50.0f };
	// regir_view mode, 0 renders normally
	i32 viewMode { 0 };
	bool viewChanged { false };
	bool clearRequested { false };
	std::vector<RegirDebugSample> samples {};
	bool samplesValid { false };
	bool focusPending { false };

	// -- gpu
	vkof::Pipeline pipeline {};
	vkof::Buffer sampleBuffer {};
};

void regir_debug_destroy(RegirDebugState & state);

// resolves the selected probe's reservoirs when a readback is due; a node
// with id 0 when nothing needs to run. schedule after integrate
[[nodiscard]] vkof::RenderNode regir_debug_node(
	Regir const & regir,
	RegirDebugState & state
);

// after the graph executed; blocking readback when one is due
void regir_debug_readback(
	Regir const & regir,
	RegirDebugState & state,
	u32 const fillIndex,
	u32 const frameIndex,
	f32 const cellSize
);

// selects the probe at slot, e.g. from the path inspector
void regir_debug_select(RegirDebugState & state, u32 const slot);

void regir_debug_imgui(RegirDebugState & state, f32v3 const cameraPosition);

// the ui's focus button; bounds of the selected probe's cell
[[nodiscard]] bool regir_debug_focus(
	RegirDebugState & state,
	f32v3 & center,
	f32 & extent
);

// draws the selected probe's cell, representative and reservoirs
void regir_debug_overlay(
	RegirDebugState const & state,
	DebugPathCamera const & camera,
	f32v2 const windowSize
);

} // namespace ponder
