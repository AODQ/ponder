#pragma once

#include <vkof/shader-cache.hpp>

#include <chrono>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace shaderd
{
	enum struct JobStage {
		idle,
		queued,
		slang,
		driver,
	};

	struct PipelineRow {
		std::string key;
		vkof::shader_cache::Variant variant;
		std::optional<vkof::shader_cache::Artifact> artifact;
		bool fresh;
		JobStage stage;
		std::chrono::steady_clock::time_point stageStart;
	};

	struct MonitorHeader {
		std::string root;
		std::string cache;
		std::string slangVersion;
		std::string stderrLog;
		bool vulkan;
		u64 generation;
		bool building;
		u32 batchDone;
		u32 batchTotal;
	};

	enum struct CommandKind {
		rebuild,
		forget,
		quit,
	};

	struct Command {
		CommandKind kind;
		std::string filter;
	};

	struct MonitorSnapshot {
		MonitorHeader header;
		std::vector<PipelineRow> rows;
		std::vector<std::string> log;
	};

	struct Monitor {
		std::mutex mutex;
		MonitorHeader header {};
		std::vector<PipelineRow> rows {};
		std::map<std::string, JobStage> stages {};
		std::map<std::string, std::chrono::steady_clock::time_point>
			stageStarts {};
		std::deque<std::string> log {};
		std::vector<Command> commands {};
		bool plain { true };
		std::function<void()> wake {};
	};

	void monitor_log(Monitor & monitor, std::string const & line);
	void monitor_stage_set(
		Monitor & monitor,
		std::string const & key,
		JobStage stage
	);
	void monitor_rows_set(Monitor & monitor, std::vector<PipelineRow> rows);
	void monitor_header_set(Monitor & monitor, MonitorHeader const & header);
	void monitor_command_push(Monitor & monitor, Command const & command);
	std::vector<Command> monitor_commands_take(Monitor & monitor);
	MonitorSnapshot monitor_snapshot(Monitor & monitor);
	void monitor_wake(Monitor & monitor);

	std::string variant_short(vkof::shader_cache::Variant const & variant);
	std::string entry_short(std::string const & entry);
	std::string statistics_format(
		vkof::shader_cache::Statistics const & statistics
	);
}
