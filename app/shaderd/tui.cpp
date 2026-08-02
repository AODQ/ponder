#include "tui.hpp"

#include <ftxui/component/app.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>

#include <algorithm>
#include <filesystem>
#include <cstdio>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace ui = ftxui;
namespace sc = vkof::shader_cache;

namespace
{
	enum struct SortMode {
		name,
		driver,
		registers,
		spill,
	};

	struct TuiState {
		size_t selected { 0u };
		SortMode sort { SortMode::name };
		bool showErrors { true };
		std::string selectedKey {};
	};
}

static constexpr int skColumnEntry = 22;
static constexpr int skColumnVariant = 24;
static constexpr int skColumnState = 14;
static constexpr int skColumnTime = 9;
static constexpr int skColumnRegisters = 6;
static constexpr int skColumnSpill = 9;
static constexpr int skColumnOccupancy = 6;
static constexpr int skColumnIsa = 9;
static constexpr int skErrorHeight = 14;

bool shaderd::stdout_is_terminal() {
#if defined(_WIN32)
	return _isatty(_fileno(stdout)) != 0;
#else
	return isatty(fileno(stdout)) != 0;
#endif
}

// -----------------------------------------------------------------------------
// -- formatting
// -----------------------------------------------------------------------------

static std::string seconds_format(f64 const milliseconds) {
	char buffer[32];
	if (milliseconds < 1000.0) {
		snprintf(buffer, sizeof(buffer), "%.0f ms", milliseconds);
	} else {
		snprintf(buffer, sizeof(buffer), "%.1f s", milliseconds / 1000.0);
	}
	return buffer;
}

static std::string bytes_format(u32 const bytes) {
	if (bytes < 1024u) { return std::to_string(bytes) + " B"; }
	char buffer[32];
	snprintf(buffer, sizeof(buffer), "%.1f K", (f64)bytes / 1024.0);
	return buffer;
}

static ui::Element cell(std::string const & text, int const width) {
	return ui::text(text) | ui::size(ui::WIDTH, ui::EQUAL, width);
}

static char const * sort_name(SortMode const sort) {
	switch (sort) {
		case SortMode::driver: return "driver time";
		case SortMode::registers: return "registers";
		case SortMode::spill: return "spills";
		default: return "name";
	}
}

// -----------------------------------------------------------------------------
// -- panes
// -----------------------------------------------------------------------------

static ui::Element header_render(shaderd::MonitorHeader const & header) {
	ui::Elements lines;
	lines.emplace_back(ui::hbox({
		ui::text("shaderd") | ui::bold,
		ui::text(
			"   slang " + header.slangVersion
			+ (header.vulkan ? "" : "   (no vulkan)")
			+ "   generation " + std::to_string(header.generation)
		),
	}));
	// -- the cache usually sits inside the root
	std::filesystem::path const relativeCache = (
		std::filesystem::path(header.cache).lexically_relative(header.root)
	);
	bool const cacheInRoot = (
		!relativeCache.empty() && *relativeCache.begin() != ".."
	);
	lines.emplace_back(
		ui::text(
			header.root + "  →  "
			+ (cacheInRoot ? relativeCache.string() : header.cache)
		) | ui::dim
	);
	if (header.building) {
		f32 const progress = (
			header.batchTotal > 0u
			? (f32)header.batchDone / (f32)header.batchTotal
			: 0.0f
		);
		lines.emplace_back(ui::hbox({
			ui::text(
				"building " + std::to_string(header.batchDone) + "/"
				+ std::to_string(header.batchTotal) + "  "
			) | ui::color(ui::Color::Cyan),
			ui::gauge(progress) | ui::color(ui::Color::Cyan) | ui::flex,
		}));
	} else {
		lines.emplace_back(ui::text("idle") | ui::dim);
	}
	if (!header.stderrLog.empty()) {
		lines.emplace_back(
			ui::text("driver/layer output → " + header.stderrLog) | ui::dim
		);
	}
	return ui::vbox(lines);
}

static ui::Element state_render(shaderd::PipelineRow const & row) {
	auto const now = std::chrono::steady_clock::now();
	f64 const elapsedMs = std::chrono::duration<f64, std::milli>(
		now - row.stageStart
	).count();
	switch (row.stage) {
		case shaderd::JobStage::queued:
			return cell("queued", skColumnState) | ui::dim;
		case shaderd::JobStage::slang:
			return (
				cell("slang " + seconds_format(elapsedMs), skColumnState)
				| ui::color(ui::Color::Cyan)
			);
		case shaderd::JobStage::driver:
			return (
				cell("driver " + seconds_format(elapsedMs), skColumnState)
				| ui::color(ui::Color::Cyan)
			);
		default: break;
	}
	if (!row.artifact) {
		return cell("missing", skColumnState) | ui::color(ui::Color::Yellow);
	}
	if (!row.artifact->ok) {
		return (
			cell("FAILED", skColumnState)
			| ui::color(ui::Color::Red)
			| ui::bold
		);
	}
	if (!row.fresh) {
		return cell("stale", skColumnState) | ui::color(ui::Color::Yellow);
	}
	return (
		cell(row.artifact->hasBinary ? "ok" : "ok (no bin)", skColumnState)
		| ui::color(ui::Color::Green)
	);
}

static ui::Element row_render(
	shaderd::PipelineRow const & row,
	bool const selected
) {
	ui::Elements cells;
	cells.emplace_back(
		cell(shaderd::entry_short(row.variant.entry), skColumnEntry)
	);
	cells.emplace_back(
		cell(shaderd::variant_short(row.variant), skColumnVariant)
	);
	cells.emplace_back(state_render(row));
	bool const hasArtifact = row.artifact.has_value() && row.artifact->ok;
	cells.emplace_back(cell(
		hasArtifact ? seconds_format(row.artifact->slangMs) : "",
		skColumnTime
	));
	cells.emplace_back(cell(
		hasArtifact ? seconds_format(row.artifact->vulkanMs) : "",
		skColumnTime
	));
	std::optional<sc::Statistics> const statistics = (
		hasArtifact ? row.artifact->statistics : std::nullopt
	);
	if (statistics) {
		cells.emplace_back(
			cell(std::to_string(statistics->registers), skColumnRegisters)
		);
		ui::Color const spillColor = (
			statistics->spillBytes >= 1024u ? ui::Color(ui::Color::Red)
			: statistics->spillBytes > 0u ? ui::Color(ui::Color::Yellow)
			: ui::Color(ui::Color::Default)
		);
		cells.emplace_back(
			cell(
				statistics->spillBytes > 0u
				? bytes_format(statistics->spillBytes)
				: "–",
				skColumnSpill
			)
			| ui::color(spillColor)
		);
		ui::Color const occupancyColor = (
			statistics->occupancyPercent < 25u ? ui::Color(ui::Color::Red)
			: statistics->occupancyPercent < 50u ? ui::Color(ui::Color::Yellow)
			: ui::Color(ui::Color::Default)
		);
		cells.emplace_back(
			cell(
				statistics->warpsPerSm > 0u
				? std::to_string(statistics->occupancyPercent) + "%"
				: "?",
				skColumnOccupancy
			)
			| ui::color(occupancyColor)
		);
		cells.emplace_back(
			cell(bytes_format(statistics->isaBytes), skColumnIsa)
		);
	}
	ui::Element line = ui::hbox(cells);
	if (selected) { line = line | ui::inverted | ui::focus; }
	return line;
}

static ui::Element table_render(
	std::vector<shaderd::PipelineRow> const & rows,
	size_t const selected
) {
	ui::Element const heading = ui::hbox({
		cell("entry", skColumnEntry),
		cell("variant", skColumnVariant),
		cell("state", skColumnState),
		cell("slang", skColumnTime),
		cell("driver", skColumnTime),
		cell("regs", skColumnRegisters),
		cell("spill", skColumnSpill),
		cell("occ", skColumnOccupancy),
		cell("isa", skColumnIsa),
	}) | ui::bold;
	ui::Elements lines;
	for (size_t i = 0u; i < rows.size(); ++i) {
		lines.emplace_back(row_render(rows[i], i == selected));
	}
	if (rows.empty()) {
		lines.emplace_back(
			ui::text("no pipelines requested yet; start a renderer") | ui::dim
		);
	}
	return ui::vbox({
		heading,
		ui::separatorLight(),
		ui::vbox(lines) | ui::yframe,
	});
}

static ui::Element log_render(std::vector<std::string> const & log) {
	ui::Elements lines;
	for (std::string const & line : log) {
		ui::Element element = ui::text(line);
		if (
			line.find(" error ") != std::string::npos
			|| line.find("  error") != std::string::npos
		) {
			element = element | ui::color(ui::Color::Red);
		} else if (line.find(" skip ") != std::string::npos) {
			element = element | ui::color(ui::Color::Yellow);
		} else if (line.find(" stderr ") != std::string::npos) {
			element = element | ui::dim;
		}
		lines.emplace_back(element);
	}
	// -- keep the newest line in view
	return (
		ui::vbox(lines)
		| ui::focusPositionRelative(0.0f, 1.0f)
		| ui::yframe
		| ui::flex
	);
}

static ui::Element error_render(shaderd::PipelineRow const & row) {
	ui::Elements lines;
	std::string const & error = row.artifact->error;
	size_t begin = 0u;
	while (begin < error.size()) {
		size_t end = error.find('\n', begin);
		if (end == std::string::npos) { end = error.size(); }
		std::string const line = error.substr(begin, end - begin);
		ui::Element element = ui::text(line);
		if (line.find("error") != std::string::npos) {
			element = element | ui::color(ui::Color::Red);
		}
		lines.emplace_back(element);
		begin = end + 1u;
	}
	return ui::window(
		ui::text(
			" " + shaderd::entry_short(row.variant.entry) + " ["
			+ shaderd::variant_short(row.variant) + "] "
		) | ui::color(ui::Color::Red),
		ui::vbox(lines) | ui::yframe
	) | ui::size(ui::HEIGHT, ui::LESS_THAN, skErrorHeight);
}

// -----------------------------------------------------------------------------
// -- loop
// -----------------------------------------------------------------------------

static void rows_sort(
	std::vector<shaderd::PipelineRow> & rows,
	SortMode const sort
) {
	auto const fnStatistic = [](
		shaderd::PipelineRow const & row,
		u32 sc::Statistics::* const field
	) -> u32 {
		bool const has = (
			row.artifact && row.artifact->ok && row.artifact->statistics
		);
		return has ? (*row.artifact->statistics).*field : 0u;
	};
	auto const fnDriverMs = [](shaderd::PipelineRow const & row) -> f64 {
		return row.artifact && row.artifact->ok ? row.artifact->vulkanMs : 0.0;
	};
	std::stable_sort(
		rows.begin(),
		rows.end(),
		[&](shaderd::PipelineRow const & a, shaderd::PipelineRow const & b) {
			switch (sort) {
				case SortMode::driver:
					return fnDriverMs(a) > fnDriverMs(b);
				case SortMode::registers:
					return (
						fnStatistic(a, &sc::Statistics::registers)
						> fnStatistic(b, &sc::Statistics::registers)
					);
				case SortMode::spill:
					return (
						fnStatistic(a, &sc::Statistics::spillBytes)
						> fnStatistic(b, &sc::Statistics::spillBytes)
					);
				default:
					return (
						a.variant.entry + shaderd::variant_short(a.variant)
						< b.variant.entry + shaderd::variant_short(b.variant)
					);
			}
		}
	);
}

void shaderd::tui_run(Monitor & monitor, std::atomic<bool> const & stop) {
	ui::App app = ui::App::Fullscreen();
	{
		std::lock_guard<std::mutex> const lock(monitor.mutex);
		monitor.wake = [&app]() { app.PostEvent(ui::Event::Custom); };
	}
	TuiState state {};
	std::vector<PipelineRow> rows;

	ui::Component const renderer = ui::Renderer([&]() {
		MonitorSnapshot snapshot = monitor_snapshot(monitor);
		rows = std::move(snapshot.rows);
		rows_sort(rows, state.sort);
		// -- the selection follows its pipeline when rows reorder
		for (size_t i = 0u; i < rows.size(); ++i) {
			if (rows[i].key == state.selectedKey) { state.selected = i; }
		}
		if (state.selected >= rows.size()) {
			state.selected = rows.empty() ? 0u : rows.size() - 1u;
		}
		state.selectedKey = rows.empty() ? "" : rows[state.selected].key;

		ui::Elements panes;
		panes.emplace_back(header_render(snapshot.header));
		panes.emplace_back(ui::separator());
		panes.emplace_back(table_render(rows, state.selected));
		bool const showError = (
			state.showErrors && !rows.empty()
			&& rows[state.selected].artifact
			&& !rows[state.selected].artifact->ok
		);
		if (showError) { panes.emplace_back(error_render(rows[state.selected])); }
		panes.emplace_back(ui::separator());
		panes.emplace_back(log_render(snapshot.log));
		panes.emplace_back(ui::separator());
		panes.emplace_back(
			ui::text(
				"↑↓ select   r rebuild   R rebuild all   s sort: "
				+ std::string(sort_name(state.sort))
				+ "   enter errors   f forget   q quit"
			) | ui::dim
		);
		return ui::vbox(panes) | ui::border;
	});

	ui::Component const component = ui::CatchEvent(
		renderer,
		[&](ui::Event const & event) {
			if (event == ui::Event::Custom) {
				if (stop) { app.Exit(); }
				return false;
			}
			if (event == ui::Event::Character('q')) {
				monitor_command_push(monitor, Command {
					.kind = CommandKind::quit,
					.filter = {},
				});
				app.Exit();
				return true;
			}
			bool const up = (
				event == ui::Event::ArrowUp || event == ui::Event::Character('k')
			);
			bool const down = (
				event == ui::Event::ArrowDown || event == ui::Event::Character('j')
			);
			if (up && state.selected > 0u) {
				--state.selected;
				state.selectedKey = rows[state.selected].key;
				return true;
			}
			if (down && state.selected + 1u < rows.size()) {
				++state.selected;
				state.selectedKey = rows[state.selected].key;
				return true;
			}
			if (event == ui::Event::Character('r') && !rows.empty()) {
				monitor_command_push(monitor, Command {
					.kind = CommandKind::rebuild,
					.filter = rows[state.selected].key,
				});
				return true;
			}
			if (event == ui::Event::Character('R')) {
				monitor_command_push(monitor, Command {
					.kind = CommandKind::rebuild,
					.filter = "all",
				});
				return true;
			}
			if (event == ui::Event::Character('f')) {
				monitor_command_push(monitor, Command {
					.kind = CommandKind::forget,
					.filter = {},
				});
				return true;
			}
			if (event == ui::Event::Character('s')) {
				state.sort = (SortMode)(((int)state.sort + 1) % 4);
				return true;
			}
			if (event == ui::Event::Return) {
				state.showErrors = !state.showErrors;
				return true;
			}
			return false;
		}
	);

	app.Loop(component);
	std::lock_guard<std::mutex> const lock(monitor.mutex);
	monitor.wake = {};
}
