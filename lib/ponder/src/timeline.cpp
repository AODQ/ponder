#include <ponder/timeline.hpp>

#include <imgui.h>

#include <algorithm>
#include <cfloat>
#include <climits>

f64 ponder::timeline_seconds(Timeline const & timeline) {
	return (f64)timeline.frame / skTimelineFramesPerSecond;
}

bool ponder::timeline_imgui(Timeline & timeline) {
	bool reset = false;
	ImGuiIO const & io = ImGui::GetIO();
	if (
		ImGui::IsKeyPressed(ImGuiKey_Space, false)
		&& !io.WantTextInput
		&& !io.WantCaptureKeyboard
	) {
		timeline = Timeline { .frame = 0u, .playing = true };
		reset = true;
	}
	if (!ImGui::Begin("timeline")) {
		ImGui::End();
		return reset;
	}
	if (imgui_media_button("##timelinereset", MediaIcon::Reset, "first frame, play")) {
		timeline = Timeline { .frame = 0u, .playing = true };
		reset = true;
	}
	ImGui::SameLine();
	if (imgui_media_button("##timelineback", MediaIcon::StepBackward, "step back")) {
		timeline.frame = timeline.frame > 0u ? timeline.frame - 1u : 0u;
		timeline.playing = false;
		reset = true;
	}
	ImGui::SameLine();
	if (
		imgui_media_button(
			"##timelineplay",
			timeline.playing ? MediaIcon::Pause : MediaIcon::Play,
			timeline.playing ? "pause" : "play from this frame"
		)
	) {
		timeline.playing = !timeline.playing;
		reset = true;
	}
	ImGui::SameLine();
	if (imgui_media_button("##timelineforward", MediaIcon::StepForward, "step forward")) {
		++timeline.frame;
		timeline.playing = false;
		reset = true;
	}
	ImGui::TextUnformatted("frame");
	i32 frame = (i32)std::min(timeline.frame, (u32)INT_MAX);
	ImGui::SetNextItemWidth(-FLT_MIN);
	if (ImGui::DragInt("##timelineframe", &frame, 1.0f, 0, INT_MAX)) {
		timeline.frame = (u32)std::max(frame, 0);
		timeline.playing = false;
		reset = true;
	}
	if (ImGui::IsItemActivated()) {
		timeline.playing = false;
		reset = true;
	}
	ImGui::End();
	return reset;
}

bool ponder::imgui_media_button(
	char const * const id,
	MediaIcon const icon,
	char const * const tooltip
) {
	f32 const size = ImGui::GetFrameHeight();
	bool const pressed = ImGui::Button(id, ImVec2(size, size));
	if (tooltip && ImGui::IsItemHovered()) {
		ImGui::SetTooltip("%s", tooltip);
	}
	ImVec2 const rectMin = ImGui::GetItemRectMin();
	ImVec2 const rectMax = ImGui::GetItemRectMax();
	ImDrawList * const draw = ImGui::GetWindowDrawList();
	ImU32 const color = ImGui::GetColorU32(ImGuiCol_Text);
	f32 const pad = size * 0.28f;
	f32 const x0 = rectMin.x + pad;
	f32 const x1 = rectMax.x - pad;
	f32 const y0 = rectMin.y + pad;
	f32 const y1 = rectMax.y - pad;
	f32 const xm = 0.5f * (x0 + x1);
	f32 const ym = 0.5f * (y0 + y1);
	f32 const bar = (x1 - x0) * 0.25f;
	switch (icon) {
		case MediaIcon::Play: {
			draw->AddTriangleFilled({ x0, y0 }, { x1, ym }, { x0, y1 }, color);
		} break;
		case MediaIcon::Pause: {
			draw->AddRectFilled({ x0, y0 }, { x0 + bar * 1.4f, y1 }, color);
			draw->AddRectFilled({ x1 - bar * 1.4f, y0 }, { x1, y1 }, color);
		} break;
		case MediaIcon::StepBackward: {
			draw->AddRectFilled({ x0, y0 }, { x0 + bar, y1 }, color);
			draw->AddTriangleFilled({ x1, y0 }, { x1, y1 }, { x0 + bar, ym }, color);
		} break;
		case MediaIcon::StepForward: {
			draw->AddTriangleFilled({ x0, y0 }, { x1 - bar, ym }, { x0, y1 }, color);
			draw->AddRectFilled({ x1 - bar, y0 }, { x1, y1 }, color);
		} break;
		case MediaIcon::Reset: {
			draw->AddRectFilled({ x0, y0 }, { x0 + bar, y1 }, color);
			draw->AddTriangleFilled({ xm, y0 }, { xm, y1 }, { x0 + bar, ym }, color);
			draw->AddTriangleFilled({ x1, y0 }, { x1, y1 }, { xm, ym }, color);
		} break;
	}
	return pressed;
}
