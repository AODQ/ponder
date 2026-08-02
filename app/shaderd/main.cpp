#include "compile.hpp"
#include "monitor.hpp"
#include "tui.hpp"

#include <vkof/shader-cache.hpp>
#include <vkof/vkof.hpp>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <malloc.h>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace sc = vkof::shader_cache;

namespace
{
	struct Options {
		fs::path root;
		fs::path cache;
		u32 jobs;
		bool vulkan;
		bool plain;
		std::string command;
		std::string argument;
	};

	struct WatchedFile {
		fs::file_time_type time;
		uintmax_t size;
		u64 hash;
	};

	struct Job {
		std::string key;
		sc::Variant variant;
		bool forced;
	};

	struct BatchState {
		std::thread thread;
		std::atomic<bool> running { false };
		std::atomic<u32> done { 0u };
		u32 total { 0u };
	};
}

static constexpr i64 skPollMs = 100;
// idle time after a batch before the slang session pool shrinks to one
static constexpr i64 skSessionIdleMs = 60000;
static constexpr i64 skRequestTimeoutMs = 10000;
static constexpr f64 skMemoryFloorGib = 4.0;
static constexpr char const * skCacheDirName = ".shader-cache";

static std::atomic<bool> sStop { false };
static std::mutex sGenerationMutex;
static shaderd::Monitor sMonitor;

// -----------------------------------------------------------------------------
// -- output
// -----------------------------------------------------------------------------

static void log_line(char const * const format, ...) {
	char buffer[2048];
	va_list args;
	va_start(args, format);
	vsnprintf(buffer, sizeof(buffer), format, args);
	va_end(args);
	shaderd::monitor_log(sMonitor, buffer);
}

static void log_block(std::string const & text) {
	size_t begin = 0u;
	while (begin < text.size()) {
		size_t end = text.find('\n', begin);
		if (end == std::string::npos) { end = text.size(); }
		shaderd::monitor_log(sMonitor, "  " + text.substr(begin, end - begin));
		begin = end + 1u;
	}
}

static std::string variant_label(sc::Variant const & variant) {
	std::string label = variant.entry + " [";
	for (std::string const & define : variant.defines) {
		label += define + " ";
	}
	label += variant.optimize ? "O1]" : "O0]";
	return label;
}

// -----------------------------------------------------------------------------
// -- options
// -----------------------------------------------------------------------------

static void usage() {
	printf(
		"usage: shaderd [--root <dir>] [--out <dir>] [--jobs <n>] [--no-vulkan]\n"
		"               [--plain]\n"
		"       shaderd [--out <dir>] --status | --errors | --rebuild [entry]\n"
		"               | --forget | --quit\n"
		"\n"
		"daemon: watches --root (default: the current directory) and builds\n"
		"the compute pipelines renderers request into --out\n"
		"(default: <root>/%s). shows a terminal ui, or plain log lines with\n"
		"--plain or when stdout is not a terminal\n",
		skCacheDirName
	);
}

static std::optional<Options> options_parse(
	int const argc,
	char ** const argv
) {
	Options options {
		.root = fs::current_path(),
		.cache = {},
		.jobs = std::clamp(std::thread::hardware_concurrency(), 1u, 8u),
		.vulkan = true,
		.plain = !shaderd::stdout_is_terminal(),
		.command = {},
		.argument = {},
	};
	for (int i = 1; i < argc; ++i) {
		std::string const arg = argv[i];
		bool const hasValue = i + 1 < argc && argv[i + 1][0] != '-';
		if (arg == "--root" && hasValue) {
			options.root = argv[++i];
		} else if (arg == "--out" && hasValue) {
			options.cache = argv[++i];
		} else if (arg == "--jobs" && hasValue) {
			options.jobs = std::max(1, atoi(argv[++i]));
		} else if (arg == "--no-vulkan") {
			options.vulkan = false;
		} else if (arg == "--plain") {
			options.plain = true;
		} else if (
			arg == "--status" || arg == "--errors" || arg == "--forget"
			|| arg == "--quit" || arg == "--rebuild"
		) {
			options.command = arg.substr(2u);
			if (arg == "--rebuild" && hasValue) {
				options.argument = argv[++i];
			}
		} else {
			return std::nullopt;
		}
	}
	std::error_code ec;
	options.root = fs::weakly_canonical(options.root, ec);
	if (options.cache.empty()) {
		options.cache = options.root / skCacheDirName;
	}
	options.cache = fs::weakly_canonical(options.cache, ec);
	return options;
}

// -----------------------------------------------------------------------------
// -- json requests
// -----------------------------------------------------------------------------

static std::string request_serialize(
	std::string const & command,
	std::string const & argument,
	std::optional<std::string> const & answer
) {
	rapidjson::StringBuffer buffer;
	rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
	writer.StartObject();
	writer.Key("cmd");
	writer.String(command.c_str());
	writer.Key("arg");
	writer.String(argument.c_str());
	writer.Key("answer");
	if (answer) {
		writer.String(answer->c_str());
	} else {
		writer.Null();
	}
	writer.EndObject();
	return buffer.GetString();
}

struct Request {
	std::string command;
	std::string argument;
	std::optional<std::string> answer;
};

static std::optional<Request> request_parse(std::string const & json) {
	rapidjson::Document d;
	d.Parse(json.c_str(), json.size());
	if (
		d.HasParseError() || !d.IsObject()
		|| !d.HasMember("cmd") || !d["cmd"].IsString()
		|| !d.HasMember("answer")
	) {
		return std::nullopt;
	}
	return Request {
		.command = d["cmd"].GetString(),
		.argument = (
			d.HasMember("arg") && d["arg"].IsString() ? d["arg"].GetString() : ""
		),
		.answer = (
			d["answer"].IsString()
			? std::optional<std::string>(d["answer"].GetString())
			: std::nullopt
		),
	};
}

static std::string random_id() {
	std::random_device device;
	u64 const id = ((u64)device() << 32u) ^ device();
	char buffer[17];
	snprintf(buffer, sizeof(buffer), "%016llx", (unsigned long long)id);
	return buffer;
}

static int client_run(Options const & options) {
	if (!sc::daemon_alive(options.cache)) {
		printf("shaderd is not running for %s\n", options.cache.string().c_str());
		return 1;
	}
	fs::path const path = (
		sc::request_dir(options.cache) / (random_id() + ".json")
	);
	sc::file_write_atomic(
		path, request_serialize(options.command, options.argument, std::nullopt)
	);
	i64 const start = sc::now_unix_ms();
	while (sc::now_unix_ms() - start < skRequestTimeoutMs) {
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		std::optional<std::string> const json = sc::file_read(path);
		std::optional<Request> const request = (
			json ? request_parse(*json) : std::nullopt
		);
		if (request && request->answer) {
			printf("%s", request->answer->c_str());
			if (!request->answer->empty() && request->answer->back() != '\n') {
				printf("\n");
			}
			std::error_code ec;
			fs::remove(path, ec);
			return 0;
		}
	}
	std::error_code ec;
	fs::remove(path, ec);
	printf("shaderd did not answer\n");
	return 1;
}

// -----------------------------------------------------------------------------
// -- daemon
// -----------------------------------------------------------------------------

struct Daemon {
	Options options;
	u64 instance { 0ull };
	std::map<std::string, WatchedFile> files {};
	std::map<std::string, sc::Variant> manifest {};
	std::set<std::string> forced {};
	shaderd::ModuleCachePtr modules {};
	std::atomic<u64> generation { 0ull };
	BatchState batch {};
	std::string slangVersion {};
	vkof::DeviceShaderCoreInfo shaderCore {};
	// files edited while a batch runs; its jobs reading them are superseded
	std::mutex changedMutex {};
	std::set<std::string> changedDuringBatch {};
	// forced jobs that were skipped, forced again in the next batch
	std::vector<std::string> reforced {};
};

static bool lock_acquire(Daemon & daemon) {
	fs::path const path = sc::lock_path(daemon.options.cache);
	FILE * const file = fopen(path.string().c_str(), "wx");
	if (file) {
		fclose(file);
	} else if (sc::daemon_alive(daemon.options.cache)) {
		printf(
			"shaderd is already running for %s\n",
			daemon.options.cache.string().c_str()
		);
		return false;
	} else {
		log_line("taking over a stale lock");
	}
	return sc::file_write_atomic(
		path,
		sc::heartbeat_serialize(sc::Heartbeat {
			.instance = daemon.instance,
			.unixMs = sc::now_unix_ms(),
		})
	);
}

static void heartbeat_write(Daemon const & daemon) {
	sc::file_write_atomic(
		sc::lock_path(daemon.options.cache),
		sc::heartbeat_serialize(sc::Heartbeat {
			.instance = daemon.instance,
			.unixMs = sc::now_unix_ms(),
		})
	);
}

static f64 memory_available_gib() {
	std::ifstream file("/proc/meminfo");
	std::string line;
	while (std::getline(file, line)) {
		unsigned long long kib = 0ull;
		if (sscanf(line.c_str(), "MemAvailable: %llu kB", &kib) == 1) {
			return (f64)kib / (1024.0 * 1024.0);
		}
	}
	return 1.0e9;
}

static bool watched_extension(fs::path const & path) {
	std::string const extension = path.extension().string();
	return (
		extension == ".slang" || extension == ".slangh"
		|| extension == ".h" || extension == ".hlsl" || extension == ".glsl"
	);
}

// rescans the root, returning the paths whose contents changed
static std::vector<std::string> files_scan(Daemon & daemon) {
	std::map<std::string, WatchedFile> next;
	std::error_code ec;
	auto it = fs::recursive_directory_iterator(
		daemon.options.root, fs::directory_options::skip_permission_denied, ec
	);
	for (; !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
		fs::directory_entry const & entry = *it;
		std::string const name = entry.path().filename().string();
		if (entry.is_directory(ec)) {
			bool const skip = (
				name.starts_with(".") || name == "bin"
				|| entry.path() == daemon.options.cache
			);
			if (skip) { it.disable_recursion_pending(); }
			continue;
		}
		if (!entry.is_regular_file(ec) || !watched_extension(entry.path())) {
			continue;
		}
		std::string const path = (
			entry.path().lexically_relative(daemon.options.root).generic_string()
		);
		WatchedFile file {
			.time = entry.last_write_time(ec),
			.size = entry.file_size(ec),
			.hash = 0ull,
		};
		auto const previous = daemon.files.find(path);
		bool const same = (
			previous != daemon.files.end()
			&& previous->second.time == file.time
			&& previous->second.size == file.size
		);
		file.hash = same ? previous->second.hash : sc::file_hash(entry.path());
		next.emplace(path, file);
	}

	std::vector<std::string> changed;
	for (auto const & [path, file] : next) {
		auto const previous = daemon.files.find(path);
		bool const differs = (
			previous == daemon.files.end()
			|| previous->second.hash != file.hash
		);
		if (differs) { changed.emplace_back(path); }
	}
	for (auto const & [path, file] : daemon.files) {
		if (!next.contains(path)) { changed.emplace_back(path); }
	}
	daemon.files = std::move(next);
	return changed;
}

// returns true if the set of requested pipelines changed
static bool manifest_scan(Daemon & daemon) {
	std::map<std::string, sc::Variant> next;
	std::error_code ec;
	fs::path const dir = sc::manifest_dir(daemon.options.cache);
	for (fs::directory_entry const & entry : fs::directory_iterator(dir, ec)) {
		if (entry.path().extension() != ".json") { continue; }
		std::optional<std::string> const json = sc::file_read(entry.path());
		std::optional<sc::Variant> const variant = (
			json ? sc::manifest_parse(*json) : std::nullopt
		);
		if (!variant) { continue; }
		next.emplace(entry.path().stem().string(), *variant);
	}
	bool const changed = (
		next.size() != daemon.manifest.size()
		|| !std::equal(
			next.begin(), next.end(), daemon.manifest.begin(),
			[](auto const & a, auto const & b) { return a.first == b.first; }
		)
	);
	daemon.manifest = std::move(next);
	return changed;
}

static shaderd::HashSnapshot snapshot_take(Daemon const & daemon) {
	shaderd::HashSnapshot snapshot;
	for (auto const & [path, file] : daemon.files) {
		snapshot.emplace(path, file.hash);
	}
	return snapshot;
}

static bool artifact_clean(
	Daemon const & daemon,
	shaderd::HashSnapshot const & snapshot,
	std::string const & key
) {
	std::optional<sc::Artifact> const artifact = (
		sc::artifact_read(daemon.options.cache, key)
	);
	if (!artifact || artifact->dependencies.empty()) { return false; }
	for (sc::Dependency const & dependency : artifact->dependencies) {
		auto const it = snapshot.find(dependency.path);
		u64 const hash = (
			it != snapshot.end()
			? it->second
			: sc::file_hash(daemon.options.root / dependency.path)
		);
		if (hash != dependency.hash) { return false; }
	}
	return true;
}

static void generation_publish(Daemon & daemon, u64 const generation) {
	std::lock_guard<std::mutex> const lock(sGenerationMutex);
	if (sc::generation_read(daemon.options.cache) >= generation) { return; }
	sc::file_write_atomic(
		sc::generation_path(daemon.options.cache), std::to_string(generation)
	);
}

enum struct JobOutcome {
	ok,
	failed,
	skipped,
};

// -----------------------------------------------------------------------------
// -- pipeline statistics
// -----------------------------------------------------------------------------

// nvidia register file per streaming multiprocessor, Kepler through Ada
static constexpr u32 skNvidiaRegistersPerSm = 65536u;
static constexpr u32 skNvidiaRegisterGranularity = 8u;
static constexpr u32 skSpirvOpExecutionMode = 16u;
static constexpr u32 skSpirvLocalSize = 17u;

// the compute workgroup size from OpExecutionMode LocalSize; 0 if absent
static u32 spirv_local_size(std::string const & spirv) {
	size_t const wordCount = spirv.size() / sizeof(u32);
	auto const fnWord = [&](size_t const i) -> u32 {
		u32 word = 0u;
		memcpy(&word, spirv.data() + i * sizeof(u32), sizeof(u32));
		return word;
	};
	// -- instructions start after the 5-word header
	for (size_t i = 5u; i < wordCount;) {
		u32 const instruction = fnWord(i);
		u32 const length = instruction >> 16u;
		u32 const opcode = instruction & 0xffffu;
		if (length == 0u || i + length > wordCount) { break; }
		bool const isLocalSize = (
			opcode == skSpirvOpExecutionMode
			&& length >= 6u
			&& fnWord(i + 2u) == skSpirvLocalSize
		);
		if (isLocalSize) {
			return fnWord(i + 3u) * fnWord(i + 4u) * fnWord(i + 5u);
		}
		i += length;
	}
	return 0u;
}

static std::optional<u64> statistic_u64(
	std::vector<vkof::PipelineStatistic> const & statistics,
	char const * const name
) {
	for (vkof::PipelineStatistic const & statistic : statistics) {
		if (statistic.name == name) {
			return std::strtoull(statistic.value.c_str(), nullptr, 10);
		}
	}
	return std::nullopt;
}

// registers, spills, memory, and the occupancy registers allow
static std::optional<sc::Statistics> statistics_measure(
	vkof::PipelineComputeBinaryBuildResult const & built,
	u32 const workgroupSize,
	vkof::DeviceShaderCoreInfo const & core
) {
	std::vector<vkof::PipelineStatistic> const & statistics = built.statistics;
	if (statistics.empty()) { return std::nullopt; }
	auto const fnU32 = [&](char const * const name) -> u32 {
		std::optional<u64> const value = statistic_u64(statistics, name);
		return value ? (u32)std::min<u64>(*value, 0xffffffffull) : 0u;
	};
	sc::Statistics result {
		.registers = fnU32("Register Count"),
		// the driver sets a flag bit above the low 32 bits of this one
		.spillBytes = (u32)(
			statistic_u64(statistics, "Local Memory Size").value_or(0ull)
			& 0xffffffffull
		),
		.sharedBytes = fnU32("Shared Memory Size"),
		.stackBytes = fnU32("Stack Size"),
		.isaBytes = fnU32("Binary Size"),
		.occupancyPercent = 0u,
		.occupancyWarps = 0u,
		.warpsPerSm = 0u,
		.extra = {},
	};

	// -- occupancy, as far as registers and the warp limit allow
	bool const canEstimate = (
		result.registers > 0u && workgroupSize > 0u
		&& built.subgroupSize > 0u && core.warpsPerSm > 0u
	);
	if (canEstimate) {
		u32 const warpsPerWorkgroup = (
			(workgroupSize + built.subgroupSize - 1u) / built.subgroupSize
		);
		u32 const registersPerThread = (
			(result.registers + skNvidiaRegisterGranularity - 1u)
			/ skNvidiaRegisterGranularity
			* skNvidiaRegisterGranularity
		);
		u32 const registersPerWorkgroup = (
			registersPerThread * built.subgroupSize * warpsPerWorkgroup
		);
		u32 const workgroupsPerSm = std::min(
			skNvidiaRegistersPerSm / registersPerWorkgroup,
			core.warpsPerSm / warpsPerWorkgroup
		);
		result.occupancyWarps = workgroupsPerSm * warpsPerWorkgroup;
		result.warpsPerSm = core.warpsPerSm;
		result.occupancyPercent = result.occupancyWarps * 100u / core.warpsPerSm;
	}

	// -- anything the driver reports that isn't measured above
	for (vkof::PipelineStatistic const & statistic : statistics) {
		bool const known = (
			statistic.name == "Register Count"
			|| statistic.name == "Local Memory Size"
			|| statistic.name == "Shared Memory Size"
			|| statistic.name == "Stack Size"
			|| statistic.name == "Binary Size"
		);
		if (known) { continue; }
		result.extra += (
			(result.extra.empty() ? "" : ", ")
			+ statistic.name + " " + statistic.value
		);
	}
	return result;
}

static bool inputs_changed(
	Daemon & daemon,
	std::vector<sc::Dependency> const & dependencies
) {
	std::lock_guard<std::mutex> const lock(daemon.changedMutex);
	return std::any_of(
		dependencies.begin(),
		dependencies.end(),
		[&](sc::Dependency const & dependency) {
			return daemon.changedDuringBatch.contains(dependency.path);
		}
	);
}

static JobOutcome job_build(
	Daemon & daemon,
	Job const & job,
	shaderd::HashSnapshot const & snapshot
) {
	fs::path const & cache = daemon.options.cache;
	std::string const label = variant_label(job.variant);

	// -- skip before starting if the last build's inputs were edited since
	std::optional<sc::Artifact> const previous = (
		sc::artifact_read(cache, job.key)
	);
	std::vector<sc::Dependency> knownInputs = (
		previous ? previous->dependencies : std::vector<sc::Dependency> {}
	);
	knownInputs.emplace_back(sc::Dependency {
		.path = job.variant.entry,
		.hash = 0ull,
	});
	if (inputs_changed(daemon, knownInputs)) {
		log_line("skip    %s (edited before it started)", label.c_str());
		return JobOutcome::skipped;
	}

	// -- slang
	auto const slangStart = std::chrono::steady_clock::now();
	shaderd::CompileResult const compiled = shaderd::compile(
		daemon.options.root, job.variant, snapshot, *daemon.modules
	);
	f64 const slangMs = std::chrono::duration<f64, std::milli>(
		std::chrono::steady_clock::now() - slangStart
	).count();
	sc::Artifact artifact {
		.generation = 0ull,
		.ok = compiled.ok,
		.error = compiled.ok ? std::string {} : compiled.diagnostics,
		.dependencies = compiled.dependencies,
		.hasBinary = false,
		.slangMs = slangMs,
		.vulkanMs = 0.0,
		.statistics = {},
	};
	// a failure with unknown dependencies retries on any change
	if (artifact.dependencies.empty()) {
		for (auto const & [path, hash] : snapshot) {
			artifact.dependencies.emplace_back(sc::Dependency {
				.path = path,
				.hash = hash,
			});
		}
	}

	// -- skip the driver compile if the inputs were edited during slang
	std::vector<sc::Dependency> const & inputs = (
		compiled.dependencies.empty() ? knownInputs : compiled.dependencies
	);
	if (inputs_changed(daemon, inputs)) {
		log_line("skip    %s (edited while compiling)", label.c_str());
		return JobOutcome::skipped;
	}
	if (compiled.ok) {
		log_line(
			"slang   %s  %.0f ms (%u modules reused, %u compiled)",
			label.c_str(), slangMs,
			compiled.modulesReused, compiled.modulesCompiled
		);
		sc::file_write_atomic(
			sc::artifact_path(cache, job.key, ".spv"), compiled.spirv
		);
	}

	// -- vulkan pipeline + binary
	std::error_code ec;
	fs::path const binaryPath = sc::artifact_path(cache, job.key, ".pbin");
	if (compiled.ok && daemon.options.vulkan) {
		bool const hasPrintf = (
			compiled.spirv.find("NonSemantic.DebugPrintf") != std::string::npos
		);
		shaderd::monitor_stage_set(
			sMonitor, job.key, shaderd::JobStage::driver
		);
		auto const vulkanStart = std::chrono::steady_clock::now();
		vkof::PipelineComputeBinaryBuildResult const built = (
			vkof::pipeline_compute_binary_build(
				srat::slice<u8 const>(
					(u8 const *)compiled.spirv.data(), compiled.spirv.size()
				),
				/*captureBinary=*/ !hasPrintf
			)
		);
		artifact.vulkanMs = std::chrono::duration<f64, std::milli>(
			std::chrono::steady_clock::now() - vulkanStart
		).count();
		if (!built.ok) {
			artifact.ok = false;
			artifact.error = "vulkan: " + built.error;
		} else if (!built.binary.empty()) {
			artifact.hasBinary = sc::file_write_atomic(
				binaryPath,
				std::string_view(
					(char const *)built.binary.data(), built.binary.size()
				)
			);
		}
		if (!artifact.hasBinary) { fs::remove(binaryPath, ec); }
		log_line(
			"vulkan  %s  %.0f ms%s%s",
			label.c_str(),
			artifact.vulkanMs,
			artifact.hasBinary ? ", binary " : "",
			artifact.hasBinary
			? (std::to_string(built.binary.size() / 1024u) + " KiB").c_str()
			: (hasPrintf ? ", no binary (uses printf)" : "")
		);
		artifact.statistics = statistics_measure(
			built, spirv_local_size(compiled.spirv), daemon.shaderCore
		);
		if (artifact.statistics) {
			log_line(
				"stats   %s  %s",
				label.c_str(),
				shaderd::statistics_format(*artifact.statistics).c_str()
			);
		}
	} else {
		fs::remove(binaryPath, ec);
	}

	// -- publish
	artifact.generation = ++daemon.generation;
	sc::file_write_atomic(
		sc::artifact_path(cache, job.key, ".json"),
		sc::artifact_serialize(artifact)
	);
	generation_publish(daemon, artifact.generation);
	if (!artifact.ok) {
		log_line("error   %s", label.c_str());
		log_block(artifact.error);
	}
	return artifact.ok ? JobOutcome::ok : JobOutcome::failed;
}

// jobs from deferredFirst on start once every job before it has finished
static void batch_run(
	Daemon & daemon,
	std::vector<Job> const jobs,
	u32 const deferredFirst,
	shaderd::HashSnapshot const snapshot
) {
	auto const start = std::chrono::steady_clock::now();
	std::atomic<u32> next { 0u };
	std::atomic<u32> end { 0u };
	std::atomic<u32> failed { 0u };
	std::atomic<u32> skipped { 0u };
	std::atomic<u32> active { 0u };
	auto const fnWorker = [&]() {
		for (u32 i = next++; i < end.load() && !sStop; i = next++) {
			// -- wait for memory while another compile is running
			while (
				memory_available_gib() < skMemoryFloorGib && active.load() > 0u
			) {
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}
			++active;
			shaderd::monitor_stage_set(
				sMonitor, jobs[i].key, shaderd::JobStage::slang
			);
			JobOutcome const outcome = job_build(daemon, jobs[i], snapshot);
			shaderd::monitor_stage_set(
				sMonitor, jobs[i].key, shaderd::JobStage::idle
			);
			failed += outcome == JobOutcome::failed ? 1u : 0u;
			skipped += outcome == JobOutcome::skipped ? 1u : 0u;
			if (outcome == JobOutcome::skipped && jobs[i].forced) {
				std::lock_guard<std::mutex> const lock(daemon.changedMutex);
				daemon.reforced.emplace_back(jobs[i].key);
			}
			--active;
			++daemon.batch.done;
		}
	};
	auto const fnPhase = [&](u32 const first, u32 const last) {
		if (first >= last || sStop) { return; }
		next = first;
		end = last;
		u32 const workerCount = std::min(daemon.options.jobs, last - first);
		std::vector<std::thread> workers;
		for (u32 i = 0u; i < workerCount; ++i) {
			workers.emplace_back(fnWorker);
		}
		for (std::thread & worker : workers) {
			worker.join();
		}
	};
	fnPhase(0u, deferredFirst);
	if (deferredFirst < jobs.size() && !sStop) {
		log_line("deferred %zu pipeline(s)", jobs.size() - deferredFirst);
	}
	fnPhase(deferredFirst, (u32)jobs.size());
	f64 const seconds = std::chrono::duration<f64>(
		std::chrono::steady_clock::now() - start
	).count();
	log_line(
		"done    generation %llu: %u ok, %u failed, %u skipped (%.1f s)",
		(unsigned long long)daemon.generation.load(),
		(u32)jobs.size() - failed.load() - skipped.load(),
		failed.load(),
		skipped.load(),
		seconds
	);
	daemon.batch.running = false;
}

static void batch_start(Daemon & daemon) {
	shaderd::HashSnapshot snapshot = snapshot_take(daemon);
	std::vector<Job> jobs;
	for (auto const & [key, variant] : daemon.manifest) {
		bool const forced = daemon.forced.contains(key);
		if (!forced && artifact_clean(daemon, snapshot, key)) { continue; }
		jobs.emplace_back(Job {
			.key = key,
			.variant = variant,
			.forced = forced,
		});
	}
	daemon.forced.clear();
	if (jobs.empty()) { return; }
	u32 const deferredFirst = (u32)(
		std::stable_partition(
			jobs.begin(),
			jobs.end(),
			[](Job const & job) { return !job.variant.deferred; }
		)
		- jobs.begin()
	);
	log_line(
		"build   %zu pipeline(s), %zu deferred",
		jobs.size(),
		jobs.size() - deferredFirst
	);
	for (Job const & job : jobs) {
		shaderd::monitor_stage_set(sMonitor, job.key, shaderd::JobStage::queued);
	}
	if (daemon.batch.thread.joinable()) { daemon.batch.thread.join(); }
	{
		std::lock_guard<std::mutex> const lock(daemon.changedMutex);
		daemon.changedDuringBatch.clear();
	}
	daemon.batch.running = true;
	daemon.batch.done = 0u;
	daemon.batch.total = (u32)jobs.size();
	daemon.batch.thread = std::thread(
		batch_run,
		std::ref(daemon),
		std::move(jobs),
		deferredFirst,
		std::move(snapshot)
	);
}

// -----------------------------------------------------------------------------
// -- control requests
// -----------------------------------------------------------------------------

static std::string answer_status(Daemon const & daemon) {
	shaderd::HashSnapshot const snapshot = snapshot_take(daemon);
	std::string text;
	text += "root        " + daemon.options.root.string() + "\n";
	text += "cache       " + daemon.options.cache.string() + "\n";
	text += "slang       " + daemon.slangVersion + "\n";
	text += std::string("vulkan      ") + (daemon.options.vulkan ? "on" : "off");
	text += "\ngeneration  " + std::to_string(daemon.generation.load()) + "\n";
	if (daemon.batch.running) {
		text += (
			"building    " + std::to_string(daemon.batch.done.load())
			+ "/" + std::to_string(daemon.batch.total) + "\n"
		);
	}
	text += "pipelines   " + std::to_string(daemon.manifest.size()) + "\n";
	for (auto const & [key, variant] : daemon.manifest) {
		std::optional<sc::Artifact> const artifact = (
			sc::artifact_read(daemon.options.cache, key)
		);
		char const * const state = (
			!artifact ? "missing"
			: !artifact->ok ? "FAILED"
			: !artifact_clean(daemon, snapshot, key) ? "stale"
			: artifact->hasBinary ? "ok+bin"
			: "ok"
		);
		char line[64];
		snprintf(line, sizeof(line), "  %-8s %s  ", state, key.c_str());
		text += line + variant_label(variant) + "\n";
		if (artifact && artifact->statistics) {
			text += (
				"           "
				+ shaderd::statistics_format(*artifact->statistics) + "\n"
			);
		}
	}
	return text;
}

static std::string answer_errors(Daemon const & daemon) {
	std::string text;
	for (auto const & [key, variant] : daemon.manifest) {
		std::optional<sc::Artifact> const artifact = (
			sc::artifact_read(daemon.options.cache, key)
		);
		if (!artifact || artifact->ok) { continue; }
		text += "-- " + variant_label(variant) + "\n" + artifact->error + "\n";
	}
	return text.empty() ? "no errors\n" : text;
}

static std::string answer_rebuild(Daemon & daemon, std::string const & filter) {
	u32 count = 0u;
	for (auto const & [key, variant] : daemon.manifest) {
		bool const match = (
			filter.empty() || filter == "all"
			|| variant.entry.find(filter) != std::string::npos
			|| key == filter
		);
		if (!match) { continue; }
		daemon.forced.insert(key);
		++count;
	}
	return "queued " + std::to_string(count) + " pipeline(s)\n";
}

static std::string answer_forget(Daemon & daemon) {
	std::error_code ec;
	u32 count = 0u;
	fs::path const dir = sc::manifest_dir(daemon.options.cache);
	for (fs::directory_entry const & entry : fs::directory_iterator(dir, ec)) {
		if (fs::remove(entry.path(), ec)) { ++count; }
	}
	daemon.manifest.clear();
	return (
		"forgot " + std::to_string(count)
		+ " pipeline(s); renderers request them again on their next start\n"
	);
}

static void requests_handle(Daemon & daemon) {
	std::error_code ec;
	fs::path const dir = sc::request_dir(daemon.options.cache);
	for (fs::directory_entry const & entry : fs::directory_iterator(dir, ec)) {
		if (entry.path().extension() != ".json") { continue; }
		std::optional<std::string> const json = sc::file_read(entry.path());
		std::optional<Request> const request = (
			json ? request_parse(*json) : std::nullopt
		);
		if (!request || request->answer) { continue; }
		std::string answer;
		if (request->command == "status") {
			answer = answer_status(daemon);
		} else if (request->command == "errors") {
			answer = answer_errors(daemon);
		} else if (request->command == "rebuild") {
			answer = answer_rebuild(daemon, request->argument);
		} else if (request->command == "forget") {
			answer = answer_forget(daemon);
		} else if (request->command == "quit") {
			answer = "stopping\n";
			sStop = true;
		} else {
			answer = "unknown command '" + request->command + "'\n";
		}
		log_line(
			"request %s %s",
			request->command.c_str(),
			request->argument.c_str()
		);
		sc::file_write_atomic(
			entry.path(),
			request_serialize(request->command, request->argument, answer)
		);
	}
}

static void on_signal(int) {
	sStop = true;
}

// -----------------------------------------------------------------------------
// -- monitor feed
// -----------------------------------------------------------------------------

namespace
{
	// lines a redirected stderr gained since the last read
	struct StderrTail {
		fs::path path {};
		u64 offset { 0ull };
		std::string partial {};
	};
}

static void monitor_publish(Daemon & daemon, StderrTail const & tail) {
	shaderd::HashSnapshot const snapshot = snapshot_take(daemon);
	std::vector<shaderd::PipelineRow> rows;
	for (auto const & [key, variant] : daemon.manifest) {
		std::optional<sc::Artifact> const artifact = (
			sc::artifact_read(daemon.options.cache, key)
		);
		rows.emplace_back(shaderd::PipelineRow {
			.key = key,
			.variant = variant,
			.artifact = artifact,
			.fresh = artifact && artifact_clean(daemon, snapshot, key),
			.stage = shaderd::JobStage::idle,
			.stageStart = {},
		});
	}
	shaderd::monitor_rows_set(sMonitor, std::move(rows));
	shaderd::monitor_header_set(sMonitor, shaderd::MonitorHeader {
		.root = daemon.options.root.string(),
		.cache = daemon.options.cache.string(),
		.slangVersion = daemon.slangVersion,
		.stderrLog = tail.path.string(),
		.vulkan = daemon.options.vulkan,
		.generation = daemon.generation.load(),
		.building = daemon.batch.running.load(),
		.batchDone = daemon.batch.done.load(),
		.batchTotal = daemon.batch.total,
	});
	shaderd::monitor_wake(sMonitor);
}

static void commands_handle(Daemon & daemon) {
	auto const fnLog = [](std::string text) {
		while (!text.empty() && text.back() == '\n') { text.pop_back(); }
		log_line("%s", text.c_str());
	};
	std::vector<shaderd::Command> const commands = (
		shaderd::monitor_commands_take(sMonitor)
	);
	for (shaderd::Command const & command : commands) {
		switch (command.kind) {
			case shaderd::CommandKind::rebuild:
				fnLog(answer_rebuild(daemon, command.filter));
				break;
			case shaderd::CommandKind::forget:
				fnLog(answer_forget(daemon));
				break;
			case shaderd::CommandKind::quit:
				sStop = true;
				break;
		}
	}
}

static void stderr_tail_read(StderrTail & tail) {
	if (tail.path.empty()) { return; }
	fflush(stderr);
	std::ifstream file(tail.path, std::ios::binary);
	if (!file.is_open()) { return; }
	file.seekg((std::streamoff)tail.offset);
	std::string const bytes(
		(std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()
	);
	tail.offset += bytes.size();
	tail.partial += bytes;
	size_t newline = tail.partial.find('\n');
	while (newline != std::string::npos) {
		std::string const line = tail.partial.substr(0u, newline);
		if (!line.empty()) { log_line("stderr  %s", line.c_str()); }
		tail.partial.erase(0u, newline + 1u);
		newline = tail.partial.find('\n');
	}
}

// -----------------------------------------------------------------------------
// -- daemon
// -----------------------------------------------------------------------------

static void poll_loop(Daemon & daemon, StderrTail & tail) {
	bool evaluate = true;
	i64 lastHeartbeat = sc::now_unix_ms();
	// 0 once the pool has been trimmed for the current idle period
	i64 lastBatchEnd = 0;
	files_scan(daemon);
	manifest_scan(daemon);
	while (!sStop) {
		i64 const now = sc::now_unix_ms();
		if (now - lastHeartbeat >= sc::skHeartbeatPeriodMs) {
			heartbeat_write(daemon);
			lastHeartbeat = now;
		}
		requests_handle(daemon);
		commands_handle(daemon);
		stderr_tail_read(tail);
		if (!daemon.forced.empty()) { evaluate = true; }

		// -- a change waits one quiet poll before building
		std::vector<std::string> const changed = files_scan(daemon);
		for (std::string const & path : changed) {
			log_line("change  %s", path.c_str());
		}
		if (daemon.batch.running && !changed.empty()) {
			std::lock_guard<std::mutex> const lock(daemon.changedMutex);
			daemon.changedDuringBatch.insert(changed.begin(), changed.end());
		}
		bool const manifestChanged = manifest_scan(daemon);
		if (manifestChanged) {
			log_line("manifest %zu pipeline(s)", daemon.manifest.size());
		}
		bool const quiet = changed.empty() && !manifestChanged;
		evaluate = evaluate || !quiet;
		if (evaluate && quiet && !daemon.batch.running) {
			evaluate = false;
			batch_start(daemon);
		}
		// -- recheck after a batch, for changes made while it ran
		if (
			!daemon.batch.running && daemon.batch.thread.joinable()
		) {
			daemon.batch.thread.join();
			std::lock_guard<std::mutex> const lock(daemon.changedMutex);
			daemon.forced.insert(daemon.reforced.begin(), daemon.reforced.end());
			daemon.reforced.clear();
			evaluate = true;
			lastBatchEnd = now;
		}
		// -- shrink the slang session pool and return freed heap when idle
		if (
			lastBatchEnd != 0
			&& !daemon.batch.running
			&& now - lastBatchEnd >= skSessionIdleMs
		) {
			lastBatchEnd = 0;
			u32 const released = shaderd::global_session_pool_trim(1u);
			malloc_trim(0);
			log_line("idle    released %u slang session(s)", released);
		}
		monitor_publish(daemon, tail);
		std::this_thread::sleep_for(std::chrono::milliseconds(skPollMs));
	}
	shaderd::monitor_wake(sMonitor);
}

static int daemon_run(Options const & options) {
	Daemon daemon {
		.options = options,
		.instance = 0ull,
	};
	daemon.instance = std::random_device {}();
	sMonitor.plain = options.plain;
	std::error_code ec;
	fs::create_directories(sc::manifest_dir(options.cache), ec);
	fs::create_directories(sc::artifact_dir(options.cache), ec);
	fs::create_directories(sc::request_dir(options.cache), ec);
	if (!lock_acquire(daemon)) { return 1; }
	std::signal(SIGINT, on_signal);
	std::signal(SIGTERM, on_signal);

	// -- driver and validation layer output would garble the ui
	StderrTail tail {};
	if (!options.plain) {
		tail.path = options.cache / "shaderd-stderr.log";
		if (!std::freopen(tail.path.string().c_str(), "w", stderr)) {
			tail.path.clear();
		}
	}

	daemon.modules = shaderd::module_cache_create();
	daemon.slangVersion = shaderd::slang_version();
	daemon.generation = sc::generation_read(options.cache);
	if (options.vulkan) {
		vkof::init_shader_compiler();
		daemon.shaderCore = vkof::device_shader_core_info();
	}
	log_line(
		"watching %s -> %s (%u jobs, slang %s, vulkan %s)",
		options.root.string().c_str(),
		options.cache.string().c_str(),
		options.jobs,
		daemon.slangVersion.c_str(),
		options.vulkan ? "on" : "off"
	);

	// -- the ui owns the main thread; the daemon polls on another
	if (options.plain) {
		poll_loop(daemon, tail);
	} else {
		monitor_publish(daemon, tail);
		std::thread poller([&]() { poll_loop(daemon, tail); });
		shaderd::tui_run(sMonitor, sStop);
		sStop = true;
		poller.join();
		sMonitor.plain = true;
	}

	if (daemon.batch.running) {
		log_line("stopping: waiting for compiles already in the driver");
	} else {
		log_line("stopping");
	}
	if (daemon.batch.thread.joinable()) { daemon.batch.thread.join(); }
	if (options.vulkan) {
		vkof::shutdown();
	}
	fs::remove(sc::lock_path(options.cache), ec);
	return 0;
}

int main(int argc, char ** argv) {
	std::optional<Options> const options = options_parse(argc, argv);
	if (!options) {
		usage();
		return 1;
	}
	if (!options->command.empty()) {
		return client_run(*options);
	}
	return daemon_run(*options);
}
