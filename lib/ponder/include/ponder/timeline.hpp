#pragma once

#include <vkof/vkof.hpp>

namespace ponder {

// the current temporal timeline information of the scene
struct Timeline {
	u32 frame { 0u };
	// false stops accumulation and advancing of frame
	bool playing { true };
};

constexpr f64 skTimelineFramesPerSecond = 60.0;

[[nodiscard]] f64 timeline_seconds(Timeline const & timeline);

// returns true when accumulation must restart at the current frame
[[nodiscard]] bool timeline_imgui(Timeline & timeline);

enum struct MediaIcon : u32 {
	Play,
	Pause,
	StepBackward,
	StepForward,
	Reset,
};

[[nodiscard]] bool imgui_media_button(
	char const * const id,
	MediaIcon const icon,
	char const * const tooltip
);

} // namespace ponder
