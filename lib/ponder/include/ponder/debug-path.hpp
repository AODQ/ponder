#pragma once

#include <vkof/vkof.hpp>

#include <imgui.h>

namespace ponder {

// mirrors types/debug-path.slang
constexpr u32 skDebugPathVertexCapacity = 64u;
constexpr u32 skDebugPathPointCapacity = 2048u;
constexpr u32 skDebugPathDimensionCapacity = 512u;
constexpr u32 skDebugPathPointChord = 0xFFFFFFFFu;
constexpr u32 skDebugPathNeeNone = 0u;
constexpr u32 skDebugPathNeeBlocked = 1u;
constexpr u32 skDebugPathNeeVisible = 2u;
constexpr u32 skDebugPathNeeStrategyNone = 0u;
constexpr u32 skDebugPathNeeStrategyPowerCdf = 1u;
constexpr u32 skDebugPathNeeStrategyRegir = 2u;

struct DebugPathVertex {
	u32 event;
	u32 bounce;
	f32 transmittance;
	f32 frequencyShift;
	f32v3 position;
	f32 emitTime;
	f32v3 direction;
	f32 pathLength;
	f32v3 throughputRgb;
	f32 rrChance;
	f32v3 emissionRgb;
	// surface vertices: the hit instance and its material index
	u32 surfaceInstance;
	f32v4 wavelengths;
	f32v3 contributionRgb;
	u32 surfaceMaterial;
	f32v4 throughputAmounts;
	f32 pdfBsdfFromHit;
	u32 isDeltaDirac;
	f32 pdfEmitFromHit;
	f32 pdfEmitFromNee;
	f32v3 directRgb;
	f32 pdfBsdfFromNee;
	f32v3 neePosition;
	u32 neeState;
	u32 regirSlot;
	u32 neeStrategy;
	u32 neeReservoir;
	u32 pad0;
};
static_assert(sizeof(DebugPathVertex) == 192u);

struct DebugPathPoint {
	f32v3 position;
	u32 kind;
};
static_assert(sizeof(DebugPathPoint) == 16u);

// one recorded path of the selected pixel
struct DebugPath {
	u32 vertexCount;
	u32 pointCount;
	u32 dimensionCount;
	u32 overflow;
	u32v2 pixel;
	u32 sampleIndex;
	// size of DebugPath as compiled into the shader
	u32 byteCount;
	DebugPathVertex vertices[skDebugPathVertexCapacity];
	DebugPathPoint points[skDebugPathPointCapacity];
	u32 dimensions[skDebugPathDimensionCapacity];
};
static_assert(sizeof(DebugPath) == 47136u);

struct DebugPathCamera {
	f32v3 position;
	f32v3 forward;
	f32 fovY;
	u32 renderWidth;
	u32 renderHeight;
};

// the pixel picker and path inspector of the viewer
struct DebugPathState {
	// the next frame records the path of pixel
	bool capturePending { false };
	// the pending capture picks the material of the first surface hit
	bool materialPickPending { false };
	// the pending capture focuses the lens on the first hit
	bool focusPickPending { false };
	// path holds a recorded path
	bool captured { false };
	u32v2 pixel { 0u, 0u };
	// the view the pixel was picked from; recaptures trace from it
	DebugPathCamera camera {};
	// -1 for none
	i32 highlightedVertex { -1 };
	DebugPath path {};
	// set by the inspector's "inspect probe", consumed by the viewer
	u32 regirInspectSlot { 0xFFFFFFFFu };
	// g + right click also clears the regir probe selection
	bool regirClearRequested { false };
};

struct DebugPathInputInfo {
	DebugPathCamera camera;
	// framebuffer image for the magnifier, linear rec.709
	ImTextureID image;
	u32 renderWidth;
	u32 renderHeight;
	f32v2 windowSize;
};

// help tooltip ('`'), magnifier ('g') and picking ('g' + click)
void debug_path_input(
	DebugPathState & state,
	DebugPathInputInfo const & info
);

// true while picking, so the camera ignores the click
[[nodiscard]] bool debug_path_picking();

// after a material pick capture is read back: the first surface hit's model
// draw index and material index; false if nothing was picked
[[nodiscard]] bool debug_path_material_pick(
	DebugPathState & state,
	i32 & modelDrawIndex,
	i32 & materialIndex
);

// after a focus pick capture is read back: the first hit's distance along the
// picked view's forward; false if nothing was hit
[[nodiscard]] bool debug_path_focus_pick(
	DebugPathState & state,
	f32 & focalDistance
);

void debug_path_imgui(DebugPathState & state);

// 'f' with a highlighted vertex; bounds of the segment leading to it
[[nodiscard]] bool debug_path_focus(
	DebugPathState const & state,
	f32v3 & center,
	f32 & extent
);

// draws the recorded path and selected pixel over the viewport
void debug_path_overlay(
	DebugPathState & state,
	DebugPathCamera const & camera,
	f32v2 const windowSize
);

// draws a world-space segment over the viewport with the path overlay's
// projection; false when it lies entirely behind the camera
bool debug_overlay_segment(
	DebugPathCamera const & camera,
	f32v2 const windowSize,
	f32v3 const a,
	f32v3 const b,
	ImU32 const color,
	f32 const thickness
);

struct DebugOverlaySphere {
	f32v3 center;
	f32 radius;
	ImU32 color;
};

// draws world-space spheres as filled circles over the viewport, skipping
// those behind the camera or off screen
void debug_overlay_spheres(
	DebugPathCamera const & camera,
	f32v2 const windowSize,
	srat::slice<DebugOverlaySphere const> const spheres
);

} // namespace ponder
