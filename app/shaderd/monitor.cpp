#include "monitor.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <filesystem>

namespace sc = vkof::shader_cache;

static constexpr size_t skLogCapacity = 2000u;

void shaderd::monitor_log(Monitor & monitor, std::string const & line) {
	std::time_t const now = std::time(nullptr);
	char stamp[16];
	std::strftime(stamp, sizeof(stamp), "%H:%M:%S", std::localtime(&now));
	std::string const stamped = std::string(stamp) + " " + line;
	{
		std::lock_guard<std::mutex> const lock(monitor.mutex);
		monitor.log.emplace_back(stamped);
		while (monitor.log.size() > skLogCapacity) { monitor.log.pop_front(); }
		if (monitor.plain) {
			printf("[shaderd %s\n", (std::string(stamp) + "] " + line).c_str());
			fflush(stdout);
		}
	}
	monitor_wake(monitor);
}

void shaderd::monitor_stage_set(
	Monitor & monitor,
	std::string const & key,
	JobStage const stage
) {
	{
		std::lock_guard<std::mutex> const lock(monitor.mutex);
		monitor.stages[key] = stage;
		monitor.stageStarts[key] = std::chrono::steady_clock::now();
	}
	monitor_wake(monitor);
}

void shaderd::monitor_rows_set(
	Monitor & monitor,
	std::vector<PipelineRow> rows
) {
	std::lock_guard<std::mutex> const lock(monitor.mutex);
	monitor.rows = std::move(rows);
}

void shaderd::monitor_header_set(
	Monitor & monitor,
	MonitorHeader const & header
) {
	std::lock_guard<std::mutex> const lock(monitor.mutex);
	monitor.header = header;
}

void shaderd::monitor_command_push(Monitor & monitor, Command const & command) {
	std::lock_guard<std::mutex> const lock(monitor.mutex);
	monitor.commands.emplace_back(command);
}

std::vector<shaderd::Command> shaderd::monitor_commands_take(
	Monitor & monitor
) {
	std::lock_guard<std::mutex> const lock(monitor.mutex);
	std::vector<Command> commands = std::move(monitor.commands);
	monitor.commands.clear();
	return commands;
}

shaderd::MonitorSnapshot shaderd::monitor_snapshot(Monitor & monitor) {
	std::lock_guard<std::mutex> const lock(monitor.mutex);
	MonitorSnapshot snapshot {
		.header = monitor.header,
		.rows = monitor.rows,
		.log = std::vector<std::string>(monitor.log.begin(), monitor.log.end()),
	};
	// -- live job stages override the rows' idle state
	for (PipelineRow & row : snapshot.rows) {
		auto const stage = monitor.stages.find(row.key);
		if (stage == monitor.stages.end()) { continue; }
		row.stage = stage->second;
		row.stageStart = monitor.stageStarts[row.key];
	}
	return snapshot;
}

// under the lock, so the ui can't be torn down mid-call
void shaderd::monitor_wake(Monitor & monitor) {
	std::lock_guard<std::mutex> const lock(monitor.mutex);
	if (monitor.wake) { monitor.wake(); }
}

// -----------------------------------------------------------------------------
// -- formatting
// -----------------------------------------------------------------------------

std::string shaderd::variant_short(sc::Variant const & variant) {
	std::string text;
	for (std::string define : variant.defines) {
		if (define.starts_with("PONDER_")) { define = define.substr(7u); }
		std::transform(
			define.begin(), define.end(), define.begin(),
			[](unsigned char const c) { return (char)std::tolower(c); }
		);
		text += (text.empty() ? "" : "+") + define;
	}
	if (text.empty()) { text = "default"; }
	if (!variant.optimize) { text += " O0"; }
	return text;
}

std::string shaderd::entry_short(std::string const & entry) {
	return std::filesystem::path(entry).stem().string();
}

std::string shaderd::statistics_format(sc::Statistics const & statistics) {
	std::string text = std::to_string(statistics.registers) + " regs";
	text += (
		statistics.spillBytes == 0u
		? ", no spills"
		: ", spills " + std::to_string(statistics.spillBytes) + " B/thread"
	);
	if (statistics.sharedBytes > 0u) {
		text += ", shared " + std::to_string(statistics.sharedBytes) + " B";
	}
	if (statistics.stackBytes > 0u) {
		text += ", stack " + std::to_string(statistics.stackBytes) + " B";
	}
	text += ", isa " + std::to_string(statistics.isaBytes / 1024u) + " KiB";
	if (statistics.warpsPerSm > 0u) {
		text += (
			", occupancy " + std::to_string(statistics.occupancyPercent) + "% ("
			+ std::to_string(statistics.occupancyWarps) + "/"
			+ std::to_string(statistics.warpsPerSm) + " warps, by registers)"
		);
	}
	if (!statistics.extra.empty()) { text += ", " + statistics.extra; }
	return text;
}
