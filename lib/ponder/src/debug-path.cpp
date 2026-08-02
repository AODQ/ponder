#include <ponder/debug-path.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <string>

namespace {

constexpr f32v3 skWorldUp = { 0.0f, 1.0f, 0.0f };
// odd so the hovered pixel is centered
constexpr i32 skMagnifierPixels = 21;
constexpr f32 skMagnifierCellSize = 10.0f;

// mirrors TransportEvent in path/transport.slang
char const * event_name(u32 const event) {
	switch (event) {
		case 0u: return "surface";
		case 1u: return "medium scatter";
		case 2u: return "escape";
		case 3u: return "pending";
		case 4u: return "lost";
		case 5u: return "emission";
		default: return "?";
	}
}

ImU32 event_color(u32 const event) {
	switch (event) {
		case 0u: return IM_COL32(80, 220, 80, 255);
		case 1u: return IM_COL32(80, 220, 220, 255);
		case 2u: return IM_COL32(140, 170, 255, 255);
		case 4u: return IM_COL32(255, 60, 60, 255);
		case 5u: return IM_COL32(255, 170, 40, 255);
		default: return IM_COL32(255, 255, 255, 255);
	}
}

// mirrors Dimension and BounceDimension in sample/sampler.slang
std::string dimension_name(u32 const dimension) {
	char const * const names[] = {
		"PixelJitterX", "PixelJitterY", "CameraShutter", "Wavelength",
		"CameraLensX", "CameraLensY",
	};
	constexpr u32 kBounceStart = 6u;
	if (dimension < kBounceStart) {
		return names[dimension];
	}
	char const * const bounceNames[] = {
		"LobeSelect",
		"DirectionX",
		"DirectionY",
		"EmitterSelect",
		"EmitterElement",
		"EmitterPositionX",
		"EmitterPositionY",
		"RussianRoulette",
		"MediumDistance",
		"RegirShade",
	};
	constexpr u32 kBounceCount = 10u;
	u32 const local = dimension - kBounceStart;
	return (
		"bounce " + std::to_string(local / kBounceCount)
		+ " " + bounceNames[local % kBounceCount]
	);
}

struct ProjectBasis {
	f32v3 position;
	f32v3 forward;
	f32v3 right;
	f32v3 up;
	f32v2 windowSize;
};

// mirrors Camera_Pinhole in util/camera.slang
ProjectBasis project_basis(
	ponder::DebugPathCamera const & camera,
	f32v2 const windowSize
) {
	f32v3 const forward = f32v3_normalize(camera.forward);
	f32v3 const right = f32v3_normalize(f32v3_cross(forward, skWorldUp));
	f32v3 const up = f32v3_cross(right, forward);
	f32 const tanHalfFovY = std::tan(camera.fovY * 0.5f);
	f32 const aspect = (f32)camera.renderWidth / (f32)camera.renderHeight;
	return ProjectBasis {
		.position = camera.position,
		.forward = forward,
		.right = right * (tanHalfFovY * aspect),
		.up = up * tanHalfFovY,
		.windowSize = windowSize,
	};
}

// camera-space depth of a point
f32 project_depth(ProjectBasis const & basis, f32v3 const point) {
	return f32v3_dot(point - basis.position, basis.forward);
}

// point must be in front of the camera
ImVec2 project_window(ProjectBasis const & basis, f32v3 const point) {
	f32v3 const d = point - basis.position;
	f32 const depth = f32v3_dot(d, basis.forward);
	f32 const ndcX = (
		f32v3_dot(d, basis.right) / (depth * f32v3_dot(basis.right, basis.right))
	);
	f32 const ndcY = (
		f32v3_dot(d, basis.up) / (depth * f32v3_dot(basis.up, basis.up))
	);
	return ImVec2(
		(ndcX + 1.0f) * 0.5f * basis.windowSize.x,
		(1.0f - ndcY) * 0.5f * basis.windowSize.y
	);
}

// false when the segment is entirely behind the camera
bool project_segment(
	ProjectBasis const & basis,
	f32v3 a,
	f32v3 b,
	ImVec2 & windowA,
	ImVec2 & windowB
) {
	constexpr f32 kNear = 1.0e-4f;
	f32 const depthA = project_depth(basis, a);
	f32 const depthB = project_depth(basis, b);
	if (depthA < kNear && depthB < kNear) {
		return false;
	}
	if (depthA < kNear) {
		a = a + (b - a) * ((kNear - depthA) / (depthB - depthA));
	} else if (depthB < kNear) {
		b = b + (a - b) * ((kNear - depthB) / (depthA - depthB));
	}
	windowA = project_window(basis, a);
	windowB = project_window(basis, b);
	return true;
}

f32 distance_to_segment(ImVec2 const p, ImVec2 const a, ImVec2 const b) {
	f32 const abX = b.x - a.x;
	f32 const abY = b.y - a.y;
	f32 const lengthSq = abX * abX + abY * abY;
	f32 const t = (
		lengthSq > 0.0f
			? std::clamp(((p.x - a.x) * abX + (p.y - a.y) * abY) / lengthSq, 0.0f, 1.0f)
			: 0.0f
	);
	f32 const dx = p.x - (a.x + abX * t);
	f32 const dy = p.y - (a.y + abY * t);
	return std::sqrt(dx * dx + dy * dy);
}

// swatch with the exact values in its tooltip
void color_cell(char const * const id, f32v3 const color) {
	f32 const height = ImGui::GetTextLineHeight();
	ImGui::ColorButton(
		id,
		ImVec4(color.x, color.y, color.z, 1.0f),
		ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float,
		ImVec2(height * 2.0f, height)
	);
}

void help_tooltip() {
	ImGui::BeginTooltip();
	ImGui::TextUnformatted("controls");
	ImGui::Separator();
	ImGui::TextUnformatted("`        hold for this help");
	ImGui::TextUnformatted("g        hold for the pixel magnifier");
	ImGui::TextUnformatted("g+click  pick a pixel to debug its path");
	ImGui::TextUnformatted("g+rclick clear the picked path and regir probe");
	ImGui::TextUnformatted("f        focus the regir probe, else the path vertex");
	ImGui::TextUnformatted("t+rclick pick a material to inspect");
	ImGui::TextUnformatted("b+click  focus the lens on a surface");
	ImGui::TextUnformatted("b+wheel  lens radius");
	ImGui::TextUnformatted("tab      orbit / first person");
	ImGui::EndTooltip();
}

void magnifier_tooltip(
	ponder::DebugPathState const & state,
	ponder::DebugPathInputInfo const & info,
	i32v2 const hovered
) {
	i32 const half = skMagnifierPixels / 2;
	f32v2 const renderSize = { (f32)info.renderWidth, (f32)info.renderHeight };
	ImVec2 const uv0 = ImVec2(
		(f32)(hovered.x - half) / renderSize.x,
		(f32)(hovered.y - half) / renderSize.y
	);
	ImVec2 const uv1 = ImVec2(
		(f32)(hovered.x + half + 1) / renderSize.x,
		(f32)(hovered.y + half + 1) / renderSize.y
	);
	f32 const extent = skMagnifierCellSize * (f32)skMagnifierPixels;
	ImGui::BeginTooltip();
	ImGui::Text("pixel %d %d", hovered.x, hovered.y);
	ImGui::Image(info.image, ImVec2(extent, extent), uv0, uv1);
	ImVec2 const origin = ImGui::GetItemRectMin();
	ImDrawList * const draw = ImGui::GetWindowDrawList();
	for (i32 i = 0; i <= skMagnifierPixels; ++i) {
		f32 const offset = (f32)i * skMagnifierCellSize;
		draw->AddLine(
			ImVec2(origin.x + offset, origin.y),
			ImVec2(origin.x + offset, origin.y + extent),
			IM_COL32(0, 0, 0, 90)
		);
		draw->AddLine(
			ImVec2(origin.x, origin.y + offset),
			ImVec2(origin.x + extent, origin.y + offset),
			IM_COL32(0, 0, 0, 90)
		);
	}
	auto const fnCell = [&](i32v2 const pixel, ImU32 const color) {
		i32 const cx = pixel.x - hovered.x + half;
		i32 const cy = pixel.y - hovered.y + half;
		if (cx < 0 || cy < 0 || cx >= skMagnifierPixels || cy >= skMagnifierPixels) {
			return;
		}
		ImVec2 const cellMin = ImVec2(
			origin.x + (f32)cx * skMagnifierCellSize,
			origin.y + (f32)cy * skMagnifierCellSize
		);
		draw->AddRect(
			cellMin,
			ImVec2(cellMin.x + skMagnifierCellSize, cellMin.y + skMagnifierCellSize),
			color,
			0.0f,
			0,
			2.0f
		);
	};
	if (state.captured) {
		fnCell(
			i32v2 { (i32)state.pixel.x, (i32)state.pixel.y },
			IM_COL32(255, 170, 40, 255)
		);
	}
	fnCell(hovered, IM_COL32(255, 255, 255, 255));
	ImGui::TextDisabled("click to pick, right click to clear");
	ImGui::EndTooltip();
}

} // namespace

bool ponder::debug_path_picking() {
	return (
		(
			ImGui::IsKeyDown(ImGuiKey_G)
			|| ImGui::IsKeyDown(ImGuiKey_T)
			|| ImGui::IsKeyDown(ImGuiKey_B)
		)
		&& !ImGui::GetIO().WantTextInput
	);
}

bool ponder::debug_path_material_pick(
	DebugPathState & state,
	i32 & modelDrawIndex,
	i32 & materialIndex
) {
	if (!state.materialPickPending || state.capturePending) {
		return false;
	}
	state.materialPickPending = false;
	// -- the pick replaces any shown path
	state.captured = false;
	DebugPath const & path = state.path;
	u32 const vertexCount = std::min(path.vertexCount, skDebugPathVertexCapacity);
	for (u32 i = 0u; i < vertexCount; ++i) {
		DebugPathVertex const & v = path.vertices[i];
		// 0 is TransportEvent::Surface
		if (v.event != 0u) { continue; }
		modelDrawIndex = (i32)v.surfaceInstance;
		materialIndex = (i32)v.surfaceMaterial;
		return true;
	}
	return false;
}

bool ponder::debug_path_focus_pick(
	DebugPathState & state,
	f32 & focalDistance
) {
	if (!state.focusPickPending || state.capturePending) {
		return false;
	}
	state.focusPickPending = false;
	// -- the pick replaces any shown path
	state.captured = false;
	DebugPath const & path = state.path;
	u32 const vertexCount = std::min(path.vertexCount, skDebugPathVertexCapacity);
	for (u32 i = 0u; i < vertexCount; ++i) {
		DebugPathVertex const & v = path.vertices[i];
		// 0 is TransportEvent::Surface, 1 MediumScatter
		if (v.event != 0u && v.event != 1u) { continue; }
		// lens offsets are perpendicular to forward, so this is the plane depth
		f32 const depth = f32v3_dot(
			v.position - state.camera.position,
			f32v3_normalize(state.camera.forward)
		);
		if (depth <= 0.0f) { return false; }
		focalDistance = depth;
		return true;
	}
	return false;
}

void ponder::debug_path_input(
	DebugPathState & state,
	DebugPathInputInfo const & info
) {
	ImGuiIO const & io = ImGui::GetIO();
	if (ImGui::IsKeyDown(ImGuiKey_GraveAccent) && !io.WantTextInput) {
		help_tooltip();
	}
	if (!debug_path_picking() || info.windowSize.x <= 0.0f) {
		return;
	}
	i32v2 const hovered = {
		std::clamp(
			(i32)(io.MousePos.x * (f32)info.renderWidth / info.windowSize.x),
			0,
			(i32)info.renderWidth - 1
		),
		std::clamp(
			(i32)(io.MousePos.y * (f32)info.renderHeight / info.windowSize.y),
			0,
			(i32)info.renderHeight - 1
		),
	};
	magnifier_tooltip(state, info, hovered);
	bool const mouseFree = !io.WantCaptureMouse;
	auto const fnCapture = [&]() {
		state.capturePending = true;
		state.captured = false;
		state.pixel = u32v2 { (u32)hovered.x, (u32)hovered.y };
		state.camera = info.camera;
		state.highlightedVertex = -1;
	};
	// -- t + right click: pick the material under the cursor
	if (ImGui::IsKeyDown(ImGuiKey_T)) {
		if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && mouseFree) {
			fnCapture();
			state.materialPickPending = true;
		}
		return;
	}
	// -- b + click: focus the lens on the surface under the cursor
	if (ImGui::IsKeyDown(ImGuiKey_B)) {
		if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mouseFree) {
			fnCapture();
			state.focusPickPending = true;
		}
		return;
	}
	// -- g + click: capture the path, g + right click: clear it
	if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mouseFree) {
		fnCapture();
	}
	if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && mouseFree) {
		state.capturePending = false;
		state.captured = false;
		state.regirClearRequested = true;
	}
}

void ponder::debug_path_imgui(DebugPathState & state) {
	if (!ImGui::Begin("debug path")) {
		ImGui::End();
		return;
	}
	if (!state.captured) {
		ImGui::TextDisabled("hold g and click to pick a pixel");
		ImGui::End();
		return;
	}
	if (ImGui::Button("capture again##debugpath")) {
		state.capturePending = true;
	}
	ImGui::SameLine();
	if (ImGui::Button("clear##debugpath")) {
		state.captured = false;
		state.highlightedVertex = -1;
		ImGui::End();
		return;
	}
	DebugPath const & path = state.path;
	if (path.byteCount != sizeof(DebugPath)) {
		ImGui::TextColored(
			ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
			"shader layout is %u bytes, viewer expects %u; restart the viewer",
			path.byteCount,
			(u32)sizeof(DebugPath)
		);
		ImGui::End();
		return;
	}
	ImGui::Text(
		"pixel %u %u  sample %u  vertices %u  points %u  dimensions %u",
		path.pixel.x,
		path.pixel.y,
		path.sampleIndex,
		path.vertexCount,
		path.pointCount,
		path.dimensionCount
	);
	if (path.overflow != 0u) {
		ImGui::TextColored(
			ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
			"overflow:%s%s%s",
			(path.overflow & 1u) ? " vertices" : "",
			(path.overflow & 2u) ? " points" : "",
			(path.overflow & 4u) ? " dimensions" : ""
		);
	}

	// -- dimensions consumed more than once
	u32 const dimensionCount = std::min(path.dimensionCount, skDebugPathDimensionCapacity);
	bool anyDuplicate = false;
	for (u32 i = 0u; i < dimensionCount; ++i) {
		for (u32 j = 0u; j < i; ++j) {
			if (path.dimensions[i] != path.dimensions[j]) {
				continue;
			}
			anyDuplicate = true;
			ImGui::TextColored(
				ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
				"dimension %u (%s) reused at draws %u and %u",
				path.dimensions[i],
				dimension_name(path.dimensions[i]).c_str(),
				j,
				i
			);
			break;
		}
	}
	if (!anyDuplicate) {
		ImGui::TextDisabled("no dimension reused");
	}
	if (ImGui::TreeNode("dimensions##debugpath")) {
		for (u32 i = 0u; i < dimensionCount; ++i) {
			ImGui::Text(
				"%3u  %4u  %s",
				i,
				path.dimensions[i],
				dimension_name(path.dimensions[i]).c_str()
			);
		}
		ImGui::TreePop();
	}

	// -- vertices
	u32 const vertexCount = std::min(path.vertexCount, skDebugPathVertexCapacity);
	std::string clipboard;
	f32v3 total = { 0.0f, 0.0f, 0.0f };
	f32 const valueColumn = ImGui::CalcTextSize("transmittance  ").x;
	auto const fnRow = [&](char const * const name) {
		f32 const x = ImGui::GetCursorPosX();
		ImGui::TextDisabled("%s", name);
		ImGui::SameLine(x + valueColumn);
	};
	for (u32 i = 0u; i < vertexCount; ++i) {
		DebugPathVertex const & v = path.vertices[i];
		total = total + v.contributionRgb;
		char line[512];
		snprintf(
			line,
			sizeof(line),
			"%u %s throughput %.4f %.4f %.4f %.4f emission %.4f %.4f %.4f "
			"contribution %.4f %.4f %.4f direct %.4f %.4f %.4f "
			"transmittance %.3f freq %.3f "
			"emit %.2f path %.2f rr %.2f "
			"pdf hit bsdf %.4f emit %.4f nee emit %.4f bsdf %.4f%s\n",
			v.bounce,
			event_name(v.event),
			v.throughputAmounts.x,
			v.throughputAmounts.y,
			v.throughputAmounts.z,
			v.throughputAmounts.w,
			v.emissionRgb.x, v.emissionRgb.y, v.emissionRgb.z,
			v.contributionRgb.x, v.contributionRgb.y, v.contributionRgb.z,
			v.directRgb.x, v.directRgb.y, v.directRgb.z,
			v.transmittance,
			v.frequencyShift,
			v.emitTime,
			v.pathLength,
			v.rrChance,
			v.pdfBsdfFromHit,
			v.pdfEmitFromHit,
			v.pdfEmitFromNee,
			v.pdfBsdfFromNee,
			(v.isDeltaDirac != 0u ? " delta" : "")
		);
		clipboard += line;

		ImGui::PushID((i32)i);
		bool const selected = state.highlightedVertex == (i32)i;
		ImGui::PushStyleColor(
			ImGuiCol_Text,
			ImGui::ColorConvertU32ToFloat4(event_color(v.event))
		);
		bool const open = ImGui::TreeNodeEx(
			"##vertex",
			(
				ImGuiTreeNodeFlags_OpenOnArrow
				| ImGuiTreeNodeFlags_SpanAvailWidth
				| (selected ? ImGuiTreeNodeFlags_Selected : 0)
			),
			"%u  %s",
			v.bounce,
			event_name(v.event)
		);
		ImGui::PopStyleColor();
		if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
			state.highlightedVertex = selected ? -1 : (i32)i;
		}
		ImGui::SameLine();
		color_cell("##contributionsummary", v.contributionRgb);
		if (v.neeState != skDebugPathNeeNone) {
			char const * const strategy = (
				v.neeStrategy == skDebugPathNeeStrategyRegir ? "regir" : "cdf"
			);
			ImGui::SameLine();
			if (v.neeState == skDebugPathNeeBlocked) {
				ImGui::TextDisabled("nee %s blocked", strategy);
			} else {
				f32 const direct = (
					f32v3_dot(v.directRgb, f32v3 { 0.2126f, 0.7152f, 0.0722f })
				);
				ImGui::Text("nee %s %.4g", strategy, direct);
			}
		}
		if (open) {
			fnRow("throughput");
			color_cell("##throughput", v.throughputRgb);
			ImGui::SameLine();
			ImGui::Text(
				"%.2f %.2f %.2f %.2f",
				v.throughputAmounts.x,
				v.throughputAmounts.y,
				v.throughputAmounts.z,
				v.throughputAmounts.w
			);
			fnRow("emission");
			color_cell("##emission", v.emissionRgb);
			fnRow("contribution");
			color_cell("##contribution", v.contributionRgb);
			fnRow("direct (nee)");
			color_cell("##direct", v.directRgb);
			fnRow("transmittance");
			color_cell(
				"##transmittance",
				f32v3 { v.transmittance, v.transmittance, v.transmittance }
			);
			fnRow("freq shift");
			ImGui::Text("%.2f", v.frequencyShift);
			fnRow("emit time");
			ImGui::Text("%.1f", v.emitTime);
			fnRow("path length");
			ImGui::Text("%.1f", v.pathLength);
			fnRow("rr");
			ImGui::Text("%.1f", v.rrChance);
			fnRow("pdf from hit");
			ImGui::Text(
				"bsdf %.4f emit %.4f%s",
				v.pdfBsdfFromHit,
				v.pdfEmitFromHit,
				(v.isDeltaDirac != 0u ? " delta" : "")
			);
			fnRow("pdf from nee");
			ImGui::Text(
				"emit %.4f bsdf %.4f", v.pdfEmitFromNee, v.pdfBsdfFromNee
			);
			fnRow("nee sample");
			char const * neeState = "none";
			if (v.neeState == skDebugPathNeeVisible) {
				neeState = "visible";
			} else if (v.neeState == skDebugPathNeeBlocked) {
				neeState = "blocked";
			}
			ImGui::TextUnformatted(neeState);
			fnRow("nee strategy");
			if (v.neeStrategy == skDebugPathNeeStrategyRegir) {
				ImGui::Text("regir reservoir %u", v.neeReservoir);
			} else if (v.neeStrategy == skDebugPathNeeStrategyPowerCdf) {
				ImGui::TextUnformatted("power cdf");
			} else {
				ImGui::TextUnformatted("none");
			}
			fnRow("regir probe");
			if (v.regirSlot == 0xFFFFFFFFu) {
				ImGui::TextUnformatted("none");
			} else {
				ImGui::Text("slot %u", v.regirSlot);
				ImGui::SameLine();
				if (ImGui::SmallButton("inspect")) {
					state.regirInspectSlot = v.regirSlot;
				}
			}
			ImGui::TreePop();
		}
		ImGui::PopID();
	}
	ImGui::Text("pixel irradiance %.4f %.4f %.4f", total.x, total.y, total.z);
	if (vertexCount > 0u) {
		DebugPathVertex const & v = path.vertices[0];
		ImGui::Text(
			"wavelengths %.1f %.1f %.1f %.1f nm",
			v.wavelengths.x,
			v.wavelengths.y,
			v.wavelengths.z,
			v.wavelengths.w
		);
	}
	if (ImGui::Button("copy##debugpath")) {
		ImGui::SetClipboardText(clipboard.c_str());
	}
	ImGui::End();
}

bool ponder::debug_path_focus(
	DebugPathState const & state,
	f32v3 & center,
	f32 & extent
) {
	if (
		!state.captured
		|| state.highlightedVertex < 0
		|| ImGui::GetIO().WantTextInput
		|| !ImGui::IsKeyPressed(ImGuiKey_F, false)
	) {
		return false;
	}
	DebugPath const & path = state.path;
	u32 const pointCount = std::min(path.pointCount, skDebugPathPointCapacity);
	f32v3 boundsMin = { FLT_MAX, FLT_MAX, FLT_MAX };
	f32v3 boundsMax = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	auto const fnInclude = [&](f32v3 const p) {
		boundsMin = {
			std::min(boundsMin.x, p.x),
			std::min(boundsMin.y, p.y),
			std::min(boundsMin.z, p.z),
		};
		boundsMax = {
			std::max(boundsMax.x, p.x),
			std::max(boundsMax.y, p.y),
			std::max(boundsMax.z, p.z),
		};
	};
	// segment i runs from point i - 1 to point i
	i32 vertex = 0;
	bool found = false;
	for (u32 i = 0u; i < pointCount; ++i) {
		if (vertex == state.highlightedVertex) {
			fnInclude(path.points[i].position);
			if (i > 0u) {
				fnInclude(path.points[i - 1u].position);
			}
			found = true;
		}
		if (path.points[i].kind != skDebugPathPointChord) {
			++vertex;
		}
	}
	if (!found) {
		return false;
	}
	center = (boundsMin + boundsMax) * 0.5f;
	extent = std::max(f32v3_length(boundsMax - boundsMin) * 0.5f, 1.0e-3f);
	return true;
}

void ponder::debug_path_overlay(
	DebugPathState & state,
	DebugPathCamera const & camera,
	f32v2 const windowSize
) {
	if ((!state.captured && !state.capturePending) || windowSize.x <= 0.0f) {
		return;
	}
	ImDrawList * const draw = ImGui::GetBackgroundDrawList();
	ProjectBasis const basis = project_basis(camera, windowSize);
	DebugPath const & path = state.path;
	u32 const pointCount = (
		state.captured ? std::min(path.pointCount, skDebugPathPointCapacity) : 0u
	);

	// -- a segment belongs to the vertex it leads to, and takes its color
	u32 kind = 3u;
	ImU32 colors[skDebugPathPointCapacity];
	i32 legs[skDebugPathPointCapacity];
	i32 vertex = 0;
	for (u32 i = 0u; i < pointCount; ++i) {
		legs[i] = vertex;
		if (path.points[i].kind != skDebugPathPointChord) {
			++vertex;
		}
	}
	for (u32 i = pointCount; i-- > 0u;) {
		if (path.points[i].kind != skDebugPathPointChord) {
			kind = path.points[i].kind;
		}
		colors[i] = event_color(kind);
	}

	// -- hover and click a leg to highlight its vertex
	ImGuiIO const & io = ImGui::GetIO();
	bool const mouseFree = !io.WantCaptureMouse && !debug_path_picking();
	i32 hoveredLeg = -1;
	f32 hoveredDistance = 6.0f;
	for (u32 i = 1u; i < pointCount && mouseFree; ++i) {
		ImVec2 a;
		ImVec2 b;
		if (
			!project_segment(
				basis,
				path.points[i - 1u].position,
				path.points[i].position,
				a,
				b
			)
		) {
			continue;
		}
		f32 const distance = distance_to_segment(io.MousePos, a, b);
		if (distance < hoveredDistance) {
			hoveredDistance = distance;
			hoveredLeg = legs[i];
		}
	}
	if (hoveredLeg >= 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
		state.highlightedVertex = (
			state.highlightedVertex == hoveredLeg ? -1 : hoveredLeg
		);
	}

	for (u32 i = 1u; i < pointCount; ++i) {
		ImVec2 a;
		ImVec2 b;
		if (
			!project_segment(
				basis,
				path.points[i - 1u].position,
				path.points[i].position,
				a,
				b
			)
		) {
			continue;
		}
		bool const highlighted = legs[i] == state.highlightedVertex;
		bool const hovered = legs[i] == hoveredLeg;
		if (highlighted) {
			draw->AddLine(a, b, IM_COL32(255, 255, 255, 255), 6.0f);
		}
		draw->AddLine(a, b, colors[i], highlighted || hovered ? 3.5f : 2.0f);
	}

	// -- nee connections, thin; yellow reached the emitter, red was blocked
	u32 const vertexCount = (
		state.captured
		? std::min(path.vertexCount, skDebugPathVertexCapacity)
		: 0u
	);
	for (u32 i = 0u; i < vertexCount; ++i) {
		DebugPathVertex const & v = path.vertices[i];
		if (v.neeState == skDebugPathNeeNone) {
			continue;
		}
		ImVec2 a;
		ImVec2 b;
		if (!project_segment(basis, v.position, v.neePosition, a, b)) {
			continue;
		}
		ImU32 const color = (
			v.neeState == skDebugPathNeeVisible
			? IM_COL32(255, 230, 60, 220)
			: IM_COL32(170, 60, 60, 170)
		);
		f32 const thickness = (
			(i32)i == state.highlightedVertex ? 2.5f : 1.0f
		);
		draw->AddLine(a, b, color, thickness);
	}

	for (u32 i = 0u; i < pointCount; ++i) {
		DebugPathPoint const & point = path.points[i];
		if (point.kind == skDebugPathPointChord) {
			continue;
		}
		if (project_depth(basis, point.position) <= 1.0e-4f) {
			continue;
		}
		draw->AddCircleFilled(
			project_window(basis, point.position),
			4.0f,
			event_color(point.kind)
		);
	}

	// -- crosshair on a newly picked pixel until its path arrives
	if (!state.capturePending || state.captured) {
		return;
	}
	ImVec2 const center = ImVec2(
		((f32)state.pixel.x + 0.5f) * windowSize.x / (f32)camera.renderWidth,
		((f32)state.pixel.y + 0.5f) * windowSize.y / (f32)camera.renderHeight
	);
	ImU32 const crosshair = IM_COL32(255, 170, 40, 255);
	draw->AddLine(
		ImVec2(center.x - 10.0f, center.y),
		ImVec2(center.x - 3.0f, center.y),
		crosshair,
		2.0f
	);
	draw->AddLine(
		ImVec2(center.x + 3.0f, center.y),
		ImVec2(center.x + 10.0f, center.y),
		crosshair,
		2.0f
	);
	draw->AddLine(
		ImVec2(center.x, center.y - 10.0f),
		ImVec2(center.x, center.y - 3.0f),
		crosshair,
		2.0f
	);
	draw->AddLine(
		ImVec2(center.x, center.y + 3.0f),
		ImVec2(center.x, center.y + 10.0f),
		crosshair,
		2.0f
	);
}

bool ponder::debug_overlay_segment(
	DebugPathCamera const & camera,
	f32v2 const windowSize,
	f32v3 const a,
	f32v3 const b,
	ImU32 const color,
	f32 const thickness
) {
	if (windowSize.x <= 0.0f) {
		return false;
	}
	ProjectBasis const basis = project_basis(camera, windowSize);
	ImVec2 windowA;
	ImVec2 windowB;
	if (!project_segment(basis, a, b, windowA, windowB)) {
		return false;
	}
	ImGui::GetBackgroundDrawList()->AddLine(windowA, windowB, color, thickness);
	return true;
}

void ponder::debug_overlay_spheres(
	DebugPathCamera const & camera,
	f32v2 const windowSize,
	srat::slice<DebugOverlaySphere const> const spheres
) {
	if (windowSize.x <= 0.0f) {
		return;
	}
	ProjectBasis const basis = project_basis(camera, windowSize);
	// basis.up is scaled by tan(fovY / 2)
	f32 const pixelsPerSlope = (
		0.5f * windowSize.y / f32v3_length(basis.up)
	);
	ImDrawList * const draw = ImGui::GetBackgroundDrawList();
	for (DebugOverlaySphere const & sphere : spheres) {
		f32 const depth = project_depth(basis, sphere.center);
		if (depth < 1.0e-4f) {
			continue;
		}
		ImVec2 const center = project_window(basis, sphere.center);
		f32 const pixelRadius = (
			std::clamp(sphere.radius / depth * pixelsPerSlope, 1.5f, 48.0f)
		);
		if (
			center.x < -pixelRadius
			|| center.y < -pixelRadius
			|| center.x > windowSize.x + pixelRadius
			|| center.y > windowSize.y + pixelRadius
		) {
			continue;
		}
		draw->AddCircleFilled(center, pixelRadius, sphere.color, 8);
	}
}
