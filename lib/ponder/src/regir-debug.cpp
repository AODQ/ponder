#include <ponder/regir-debug.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <set>
#include <string>

namespace {

static constexpr char const * skPonderShaderNewDir = PONDER_SHADER_NEW_DIR;
static constexpr u32 skProbeHistogramBins = 32u;

// mirrors the regir debug entry's stage push constants (at offset 128)
struct RegirDebugPc {
	u64 outputVa;
	u32 slot;
	u32 pad0;
};
static_assert(sizeof(RegirDebugPc) == 16u);

template <typename T>
std::vector<T> download(vkof::Buffer const buffer, u32 const count) {
	std::vector<T> result(count);
	vkof::buffer_download({
		.buffer = buffer,
		.byteOffset = 0u,
		.dst = srat::slice<u8>(
			reinterpret_cast<u8 *>(result.data()),
			(u64)count * sizeof(T)
		),
	});
	return result;
}

ponder::RegirDebugCell const * selected_cell(
	ponder::RegirDebugState const & state
) {
	if (state.selectedSlot == ponder::skWavefrontRegirSlotInvalid) {
		return nullptr;
	}
	for (ponder::RegirDebugCell const & c : state.cells) {
		if (c.slot == state.selectedSlot) {
			return &c;
		}
	}
	return nullptr;
}

char const * cell_state_name(
	ponder::RegirCell const & cell,
	u32 const fillIndex
) {
	if (cell.lastFillIndex == fillIndex) {
		return "filled";
	}
	if (cell.lastFillIndex == ponder::skWavefrontRegirSlotInvalid) {
		return "unfilled";
	}
	return "stale";
}

// blue for the weakest reservoir weight, yellow for the strongest
void legend_line(char const * const label, ImVec4 const color) {
	ImGui::ColorButton(
		label,
		color,
		ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoPicker,
		ImVec2(12.0f, 12.0f)
	);
	ImGui::SameLine();
	ImGui::TextUnformatted(label);
}

ImU32 weight_color(f32 const weight, f32 const maxWeight) {
	f32 const t = (
		maxWeight > 0.0f
			? std::clamp(std::sqrt(weight / maxWeight), 0.0f, 1.0f)
			: 0.0f
	);
	return IM_COL32(
		(int)(255.0f * t),
		(int)(160.0f + 95.0f * t),
		(int)(255.0f * (1.0f - t)),
		220
	);
}

} // namespace

void ponder::regir_debug_destroy(RegirDebugState & state) {
	if (state.pipeline.id) {
		vkof::pipeline_destroy(state.pipeline);
	}
	if (state.sampleBuffer.id) {
		vkof::buffer_destroy(state.sampleBuffer);
	}
	state = {};
}

vkof::RenderNode ponder::regir_debug_node(
	Regir const & regir,
	RegirDebugState & state
) {
	++state.framesSinceRefresh;
	state.refreshDue = (
		state.refreshRequested
		|| (
			state.autoRefresh
			&& state.framesSinceRefresh >= (u32)std::max(state.refreshInterval, 1)
		)
	);
	state.samplesScheduled = false;
	if (!state.refreshDue || state.selectedSlot == skWavefrontRegirSlotInvalid) {
		return vkof::RenderNode { 0u };
	}
	// -- compiled on first use, like the debug-path integrate variants
	if (!state.pipeline.id) {
		std::string const path = (
			std::string(skPonderShaderNewDir) + "entry/regir-debug.slang"
		);
		char const * const includePaths[] = { skPonderShaderNewDir };
		state.pipeline = (
			vkof::pipeline_compute_create({
				.pathCompute = path.c_str(),
				.defines = {},
				.includePaths = srat::slice { includePaths, 1u },
			})
		);
	}
	if (!state.sampleBuffer.id) {
		state.sampleBuffer = (
			vkof::buffer_create({
				.byteCount = sizeof(RegirDebugSample) * skWavefrontRegirReservoirCount,
				.memory = vkof::BufferMemory::DeviceOnly,
			})
		);
	}
	(void)regir;
	auto const node = (
		vkof::render_node_create({
			.queue = vkof::CommandQueue::graphics,
			.debugName = "regir debug",
		})
	);
	vkof::Pipeline const pipeline = state.pipeline;
	auto const pc = RegirDebugPc {
		.outputVa = vkof::buffer_virtual_address(state.sampleBuffer),
		.slot = state.selectedSlot,
		.pad0 = 0u,
	};
	vkof::render_node_callback({
		.node = node,
		.callback = [=](vkof::CommandBuffer const & cmd) {
			vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
				.cmd = cmd,
				.pipeline = pipeline,
				.push = pc,
				.threadgroupSize = u32v3 { skWavefrontRegirReservoirCount, 1u, 1u },
				.invocationCount = u32v3 { skWavefrontRegirReservoirCount, 1u, 1u },
			});
		},
	});
	state.samplesScheduled = true;
	return node;
}

void ponder::regir_debug_readback(
	Regir const & regir,
	RegirDebugState & state,
	u32 const fillIndex,
	u32 const frameIndex,
	f32 const cellSize
) {
	if (!state.refreshDue || regir.cellCount == 0u) {
		return;
	}
	state.refreshDue = false;
	state.refreshRequested = false;
	state.framesSinceRefresh = 0u;

	u32 const cellCount = regir.cellCount;
	u32 const mask = cellCount - 1u;
	std::vector<u32> const guards = (
		download<u32>(regir.guardBuffer, cellCount)
	);
	std::vector<RegirCell> const cells = (
		download<RegirCell>(regir.cellBuffer, cellCount)
	);

	// -- statistics over every claimed slot
	state.valid = true;
	state.fillIndex = fillIndex;
	state.cellSize = cellSize;
	state.cellCount = cellCount;
	state.claimedCount = 0u;
	state.filledCount = 0u;
	state.unfilledCount = 0u;
	state.staleCount = 0u;
	state.zeroNormCount = 0u;
	state.maxProbe = 0u;
	state.evictedCount = 0u;
	state.maxAge = 0u;
	state.probeHistogram.assign(skProbeHistogramBins, 0.0f);
	state.cells.clear();
	u64 probeSum = 0u;
	u64 ageSum = 0u;
	for (u32 slot = 0u; slot < cellCount; ++slot) {
		u32 const guard = guards[slot];
		if (guard == 0u) {
			continue;
		}
		if (guard == skWavefrontRegirSlotEvicted) {
			++state.evictedCount;
			continue;
		}
		RegirCell const & cell = cells[slot];
		u32 const age = frameIndex - cell.frameIndexLastUsed;
		ageSum += age;
		state.maxAge = std::max(state.maxAge, age);
		// the guard is the key's hash, and its home slot is hash & mask
		u32 const probeLength = (slot - (guard & mask)) & mask;
		++state.claimedCount;
		probeSum += probeLength;
		state.maxProbe = std::max(state.maxProbe, probeLength);
		state.probeHistogram[std::min(probeLength, skProbeHistogramBins - 1u)] += 1.0f;
		if (cell.lastFillIndex == fillIndex) {
			++state.filledCount;
			if (!(cell.normalizationNoncanonical > 0.0f)) {
				++state.zeroNormCount;
			}
		} else if (cell.lastFillIndex == skWavefrontRegirSlotInvalid) {
			++state.unfilledCount;
		} else {
			++state.staleCount;
		}
		state.cells.emplace_back(RegirDebugCell {
			.slot = slot,
			.probeLength = probeLength,
			.cell = cell,
		});
	}
	state.meanProbe = (
		state.claimedCount > 0u
			? (f32)((f64)probeSum / (f64)state.claimedCount)
			: 0.0f
	);
	state.meanAge = (
		state.claimedCount > 0u
			? (f32)((f64)ageSum / (f64)state.claimedCount)
			: 0.0f
	);

	// -- the selected probe's reservoirs, if regir_debug_node resolved them
	state.samplesValid = false;
	if (state.samplesScheduled && state.sampleBuffer.id) {
		state.samples = (
			download<RegirDebugSample>(state.sampleBuffer, skWavefrontRegirReservoirCount)
		);
		state.samplesValid = true;
	}
	state.samplesScheduled = false;
}

void ponder::regir_debug_select(RegirDebugState & state, u32 const slot) {
	if (state.selectedSlot == slot) {
		return;
	}
	state.selectedSlot = slot;
	state.highlightedReservoir = -1;
	state.samplesValid = false;
	state.refreshRequested = true;
}

void ponder::regir_debug_imgui(
	RegirDebugState & state,
	f32v3 const cameraPosition
) {
	if (!ImGui::Begin("regir")) {
		ImGui::End();
		return;
	}

	// -- refresh controls
	ImGui::Checkbox("auto refresh", &state.autoRefresh);
	ImGui::SameLine();
	ImGui::SetNextItemWidth(120.0f);
	ImGui::SliderInt("frames", &state.refreshInterval, 1, 240);
	ImGui::SameLine();
	if (ImGui::Button("refresh")) {
		state.refreshRequested = true;
	}
	ImGui::SameLine();
	if (ImGui::Button("clear probes")) {
		state.clearRequested = true;
	}
	char const * const viewModes[] = { "off", "probe state", "probe cells" };
	state.viewChanged |= ImGui::Combo("pixel view", &state.viewMode, viewModes, 3);
	if (state.viewMode != 0) {
		if (state.viewMode == 2) {
			legend_line("usable (one color per probe)", ImVec4(0.6f, 0.4f, 0.9f, 1.0f));
		} else {
			legend_line("usable", ImVec4(0.1f, 0.75f, 0.2f, 1.0f));
		}
		legend_line("filled, Norm 0", ImVec4(0.05f, 0.25f, 0.05f, 1.0f));
		legend_line("claimed, unfilled", ImVec4(0.8f, 0.7f, 0.05f, 1.0f));
		legend_line("stale", ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
		legend_line("no probe", ImVec4(0.8f, 0.05f, 0.05f, 1.0f));
	}
	if (!state.valid) {
		ImGui::TextUnformatted("no readback yet");
		ImGui::End();
		return;
	}

	// -- grid statistics
	ImGui::SeparatorText("grid");
	ImGui::Text(
		"claimed %u / %u (%.1f%% load)",
		state.claimedCount,
		state.cellCount,
		100.0f * (f32)state.claimedCount / (f32)std::max(state.cellCount, 1u)
	);
	ImGui::Text(
		"filled %u  unfilled %u  stale %u",
		state.filledCount,
		state.unfilledCount,
		state.staleCount
	);
	ImGui::Text("filled with Norm 0: %u", state.zeroNormCount);
	ImGui::Text(
		"evicted %u  age: mean %.1f  max %u frames",
		state.evictedCount,
		state.meanAge,
		state.maxAge
	);
	ImGui::Text(
		"probe length: mean %.2f  max %u (cap 32)",
		state.meanProbe,
		state.maxProbe
	);
	ImGui::PlotHistogram(
		"##probe",
		state.probeHistogram.data(),
		(int)state.probeHistogram.size(),
		0,
		"probe length histogram",
		0.0f,
		FLT_MAX,
		ImVec2(0.0f, 60.0f)
	);

	// -- probe list
	ImGui::SeparatorText("probes");
	if (ImGui::Button("nearest to camera") && !state.cells.empty()) {
		f32 bestDistance = FLT_MAX;
		u32 bestSlot = skWavefrontRegirSlotInvalid;
		for (RegirDebugCell const & c : state.cells) {
			f32v3 const d = c.cell.representativeOri - cameraPosition;
			f32 const distance = f32v3_dot(d, d);
			if (distance < bestDistance) {
				bestDistance = distance;
				bestSlot = c.slot;
			}
		}
		regir_debug_select(state, bestSlot);
	}
	ImGui::SameLine();
	ImGui::Checkbox("draw", &state.drawSelected);
	ImGui::Checkbox("draw all probes", &state.drawAllProbes);
	if (state.drawAllProbes) {
		ImGui::SameLine();
		ImGui::SetNextItemWidth(120.0f);
		ImGui::SliderFloat(
			"max distance",
			&state.drawAllMaxDistance,
			1.0f,
			500.0f,
			"%.0f",
			ImGuiSliderFlags_Logarithmic
		);
		legend_line("filled", ImVec4(0.31f, 0.9f, 0.43f, 1.0f));
		legend_line("filled, Norm 0", ImVec4(1.0f, 0.27f, 0.27f, 1.0f));
		legend_line("claimed, unfilled", ImVec4(1.0f, 0.86f, 0.24f, 1.0f));
		legend_line("stale", ImVec4(0.55f, 0.55f, 0.55f, 1.0f));
		legend_line("medium", ImVec4(0.31f, 0.55f, 1.0f, 1.0f));
		legend_line("selected", ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
	}
	ImGui::SameLine();
	if (ImGui::Button("focus")) {
		state.focusPending = true;
	}
	if (
		ImGui::BeginChild(
			"##probes",
			ImVec2(0.0f, 140.0f),
			ImGuiChildFlags_Borders
		)
	) {
		ImGuiListClipper clipper;
		clipper.Begin((int)state.cells.size());
		while (clipper.Step()) {
			for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
				RegirDebugCell const & c = state.cells[(size_t)i];
				std::string const label = (
					std::to_string(c.slot)
					+ "  (" + std::to_string(c.cell.key.gridCoord.x)
					+ ", " + std::to_string(c.cell.key.gridCoord.y)
					+ ", " + std::to_string(c.cell.key.gridCoord.z)
					+ ")  b" + std::to_string(c.cell.key.normalBucket)
					+ "  " + cell_state_name(c.cell, state.fillIndex)
				);
				if (
					ImGui::Selectable(
						label.c_str(),
						c.slot == state.selectedSlot
					)
				) {
					regir_debug_select(state, c.slot);
				}
			}
		}
	}
	ImGui::EndChild();

	// -- selected probe
	RegirDebugCell const * const selected = selected_cell(state);
	if (selected == nullptr) {
		ImGui::TextUnformatted("no probe selected");
		ImGui::End();
		return;
	}
	RegirCell const & cell = selected->cell;
	ImGui::SeparatorText("selected probe");
	ImGui::Text(
		"slot %u  probe %u  %s",
		selected->slot,
		selected->probeLength,
		cell_state_name(cell, state.fillIndex)
	);
	ImGui::Text(
		"key (%d, %d, %d)  bucket %u%s",
		cell.key.gridCoord.x,
		cell.key.gridCoord.y,
		cell.key.gridCoord.z,
		cell.key.normalBucket,
		cell.key.normalBucket == skWavefrontRegirNormalBucketNone ? " (medium)" : ""
	);
	ImGui::Text(
		"representative (%.3f, %.3f, %.3f)",
		cell.representativeOri.x,
		cell.representativeOri.y,
		cell.representativeOri.z
	);
	ImGui::Text(
		"normal (%.3f, %.3f, %.3f)",
		cell.representativeNor.x,
		cell.representativeNor.y,
		cell.representativeNor.z
	);
	ImGui::Text(
		"Norm %.4g  canonical %.4g  last fill %u",
		cell.normalizationNoncanonical,
		cell.normalizationCanonical,
		cell.lastFillIndex
	);
	if (!state.samplesValid) {
		ImGui::TextUnformatted("reservoirs: waiting for the next readback");
		ImGui::End();
		return;
	}

	// -- reservoir summary
	u32 validCount = 0u;
	f32 weightMin = FLT_MAX;
	f32 weightMax = 0.0f;
	f64 weightSum = 0.0;
	std::set<u32> lights;
	for (RegirDebugSample const & s : state.samples) {
		if (s.valid == 0u) {
			continue;
		}
		++validCount;
		weightMin = std::min(weightMin, s.weight);
		weightMax = std::max(weightMax, s.weight);
		weightSum += s.weight;
		lights.insert(s.lightIndex);
	}
	ImGui::Text(
		"reservoirs: %u / %u valid, %zu distinct lights",
		validCount,
		skWavefrontRegirReservoirCount,
		lights.size()
	);
	if (validCount > 0u) {
		ImGui::Text(
			"W min %.4g  mean %.4g  max %.4g",
			weightMin,
			(f32)(weightSum / (f64)validCount),
			weightMax
		);
	}

	// -- reservoir table; hovering a row highlights its line
	state.highlightedReservoir = -1;
	ImGuiTableFlags const tableFlags = (
		ImGuiTableFlags_RowBg
		| ImGuiTableFlags_ScrollY
		| ImGuiTableFlags_BordersInnerV
	);
	if (ImGui::BeginTable("##reservoirs", 5, tableFlags, ImVec2(0.0f, 200.0f))) {
		ImGui::TableSetupScrollFreeze(0, 1);
		ImGui::TableSetupColumn("#");
		ImGui::TableSetupColumn("light");
		ImGui::TableSetupColumn("W");
		ImGui::TableSetupColumn("h");
		ImGui::TableSetupColumn("p area");
		ImGui::TableHeadersRow();
		for (u32 i = 0u; i < (u32)state.samples.size(); ++i) {
			RegirDebugSample const & s = state.samples[i];
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			ImGui::Text("%u", i);
			if (ImGui::IsItemHovered()) {
				state.highlightedReservoir = (i32)i;
			}
			ImGui::TableNextColumn();
			if (s.valid == 0u) {
				ImGui::TextDisabled("empty");
				continue;
			}
			ImGui::Text("%u", s.lightIndex);
			ImGui::TableNextColumn();
			ImGui::Text("%.4g", s.weight);
			ImGui::TableNextColumn();
			ImGui::Text("%.4g", s.targetValue);
			ImGui::TableNextColumn();
			ImGui::Text("%.4g", s.pdfArea);
		}
		ImGui::EndTable();
	}
	ImGui::End();
}

bool ponder::regir_debug_focus(
	RegirDebugState & state,
	f32v3 & center,
	f32 & extent
) {
	bool const keyPressed = (
		!ImGui::GetIO().WantTextInput
		&& ImGui::IsKeyPressed(ImGuiKey_F, false)
	);
	if (!state.focusPending && !keyPressed) {
		return false;
	}
	state.focusPending = false;
	RegirDebugCell const * const selected = selected_cell(state);
	if (selected == nullptr) {
		return false;
	}
	center = selected->cell.representativeOri;
	extent = std::max(state.cellSize * 2.0f, 1.0e-3f);
	return true;
}

void ponder::regir_debug_overlay(
	RegirDebugState const & state,
	DebugPathCamera const & camera,
	f32v2 const windowSize
) {
	if (state.drawAllProbes) {
		f32 const maxDistanceSquared = (
			state.drawAllMaxDistance * state.drawAllMaxDistance
		);
		std::vector<DebugOverlaySphere> spheres;
		spheres.reserve(state.cells.size());
		for (RegirDebugCell const & c : state.cells) {
			f32v3 const d = c.cell.representativeOri - camera.position;
			if (f32v3_dot(d, d) > maxDistanceSquared) {
				continue;
			}
			ImU32 color = IM_COL32(140, 140, 140, 200);
			if (c.cell.key.normalBucket == skWavefrontRegirNormalBucketNone) {
				color = IM_COL32(80, 140, 255, 220);
			} else if (c.cell.lastFillIndex == state.fillIndex) {
				color = (
					c.cell.normalizationNoncanonical > 0.0f
						? IM_COL32(80, 230, 110, 220)
						: IM_COL32(255, 70, 70, 220)
				);
			} else if (c.cell.lastFillIndex == skWavefrontRegirSlotInvalid) {
				color = IM_COL32(255, 220, 60, 220);
			}
			if (c.slot == state.selectedSlot) {
				color = IM_COL32(255, 255, 255, 255);
			}
			spheres.emplace_back(DebugOverlaySphere {
				.center = c.cell.representativeOri,
				.radius = state.cellSize * 0.1f,
				.color = color,
			});
		}
		debug_overlay_spheres(
			camera,
			windowSize,
			srat::slice<DebugOverlaySphere const>(spheres.data(), spheres.size())
		);
	}
	if (!state.drawSelected) {
		return;
	}
	RegirDebugCell const * const selected = selected_cell(state);
	if (selected == nullptr) {
		return;
	}
	RegirCell const & cell = selected->cell;
	auto const fnLine = [&](f32v3 const a, f32v3 const b, ImU32 const color, f32 const thickness) {
		(void)debug_overlay_segment(camera, windowSize, a, b, color, thickness);
	};

	// -- the cell's box, floor(position / cellSize) * cellSize upward
	f32 const s = state.cellSize;
	f32v3 const lo = {
		(f32)cell.key.gridCoord.x * s,
		(f32)cell.key.gridCoord.y * s,
		(f32)cell.key.gridCoord.z * s,
	};
	ImU32 const boxColor = IM_COL32(80, 220, 255, 200);
	for (u32 e = 0u; e < 12u; ++e) {
		// 4 edges along each axis, offset by the other two axes' corners
		u32 const axis = e / 4u;
		u32 const corner = e % 4u;
		f32v3 a = lo;
		f32 const u = (corner & 1u) ? s : 0.0f;
		f32 const v = (corner & 2u) ? s : 0.0f;
		if (axis == 0u) {
			a = a + f32v3 { 0.0f, u, v };
		} else if (axis == 1u) {
			a = a + f32v3 { u, 0.0f, v };
		} else {
			a = a + f32v3 { u, v, 0.0f };
		}
		f32v3 const b = (
			a + f32v3 {
				axis == 0u ? s : 0.0f,
				axis == 1u ? s : 0.0f,
				axis == 2u ? s : 0.0f,
			}
		);
		fnLine(a, b, boxColor, 1.5f);
	}

	// -- representative point and normal
	f32v3 const p = cell.representativeOri;
	f32 const mark = s * 0.08f;
	ImU32 const repColor = IM_COL32(255, 255, 255, 255);
	fnLine(p - f32v3 { mark, 0.0f, 0.0f }, p + f32v3 { mark, 0.0f, 0.0f }, repColor, 2.0f);
	fnLine(p - f32v3 { 0.0f, mark, 0.0f }, p + f32v3 { 0.0f, mark, 0.0f }, repColor, 2.0f);
	fnLine(p - f32v3 { 0.0f, 0.0f, mark }, p + f32v3 { 0.0f, 0.0f, mark }, repColor, 2.0f);
	if (cell.key.normalBucket != skWavefrontRegirNormalBucketNone) {
		fnLine(p, p + cell.representativeNor * (s * 0.5f), IM_COL32(255, 120, 200, 255), 2.0f);
	}

	// -- one line per reservoir to its light point, colored by W
	if (!state.samplesValid) {
		return;
	}
	f32 maxWeight = 0.0f;
	for (RegirDebugSample const & sample : state.samples) {
		if (sample.valid != 0u) {
			maxWeight = std::max(maxWeight, sample.weight);
		}
	}
	for (u32 i = 0u; i < (u32)state.samples.size(); ++i) {
		RegirDebugSample const & sample = state.samples[i];
		if (sample.valid == 0u) {
			continue;
		}
		bool const highlighted = (i32)i == state.highlightedReservoir;
		ImU32 const color = (
			highlighted
				? IM_COL32(255, 60, 60, 255)
				: weight_color(sample.weight, maxWeight)
		);
		fnLine(p, sample.position, color, highlighted ? 3.0f : 1.0f);
		// small cross at the light point
		f32 const tick = s * 0.04f;
		fnLine(
			sample.position - f32v3 { tick, 0.0f, 0.0f },
			sample.position + f32v3 { tick, 0.0f, 0.0f },
			color,
			2.0f
		);
		fnLine(
			sample.position - f32v3 { 0.0f, 0.0f, tick },
			sample.position + f32v3 { 0.0f, 0.0f, tick },
			color,
			2.0f
		);
	}
}
