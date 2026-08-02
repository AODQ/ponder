#include <vkof/shader-cache.hpp>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;
namespace sc = vkof::shader_cache;

using JsonWriter = rapidjson::Writer<rapidjson::StringBuffer>;

static constexpr u64 skFnvOffset = 0xcbf29ce484222325ull;
static constexpr u64 skFnvPrime = 0x100000001b3ull;

static std::string hex64(u64 const value) {
	char buffer[17];
	snprintf(buffer, sizeof(buffer), "%016llx", (unsigned long long)value);
	return buffer;
}

static std::optional<u64> hex64_parse(char const * const text) {
	unsigned long long value = 0ull;
	if (sscanf(text, "%llx", &value) != 1) { return std::nullopt; }
	return (u64)value;
}

static std::optional<rapidjson::Document> json_parse(std::string const & json) {
	rapidjson::Document document;
	document.Parse(json.c_str(), json.size());
	if (document.HasParseError() || !document.IsObject()) {
		return std::nullopt;
	}
	return document;
}

// -----------------------------------------------------------------------------
// -- key
// -----------------------------------------------------------------------------

sc::Variant sc::variant_normalized(Variant variant) {
	std::replace(variant.entry.begin(), variant.entry.end(), '\\', '/');
	std::sort(variant.defines.begin(), variant.defines.end());
	variant.defines.erase(
		std::unique(variant.defines.begin(), variant.defines.end()),
		variant.defines.end()
	);
	return variant;
}

std::string sc::key(Variant const & variant) {
	Variant const normalized = variant_normalized(variant);
	std::string text = normalized.entry + "\n";
	for (std::string const & define : normalized.defines) {
		text += define + ";";
	}
	text += normalized.optimize ? "\nO1" : "\nO0";
	return hex64(bytes_hash(text));
}

// -----------------------------------------------------------------------------
// -- files
// -----------------------------------------------------------------------------

u64 sc::bytes_hash(std::string_view const bytes) {
	u64 hash = skFnvOffset;
	for (char const c : bytes) {
		hash ^= (u8)c;
		hash *= skFnvPrime;
	}
	return hash;
}

u64 sc::file_hash(fs::path const & path) {
	std::optional<std::string> const bytes = file_read(path);
	return bytes ? bytes_hash(*bytes) : 0ull;
}

std::optional<std::string> sc::file_read(fs::path const & path) {
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open()) { return std::nullopt; }
	std::ostringstream stream;
	stream << file.rdbuf();
	return stream.str();
}

bool sc::file_write_atomic(
	fs::path const & path,
	std::string_view const bytes
) {
	std::error_code ec;
	fs::create_directories(path.parent_path(), ec);
	fs::path const temporary = path.string() + ".tmp";
	{
		std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
		if (!file.is_open()) { return false; }
		file.write(bytes.data(), (std::streamsize)bytes.size());
		if (!file) { return false; }
	}
	fs::rename(temporary, path, ec);
	return !ec;
}

i64 sc::now_unix_ms() {
	return (
		std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::system_clock::now().time_since_epoch()
		).count()
	);
}

// -----------------------------------------------------------------------------
// -- layout
// -----------------------------------------------------------------------------

fs::path sc::lock_path(fs::path const & cache) {
	return cache / "shaderd.lock";
}

fs::path sc::generation_path(fs::path const & cache) {
	return cache / "generation";
}

fs::path sc::manifest_dir(fs::path const & cache) {
	return cache / "manifest";
}

fs::path sc::artifact_dir(fs::path const & cache) {
	return cache / "artifacts";
}

fs::path sc::request_dir(fs::path const & cache) {
	return cache / "requests";
}

fs::path sc::manifest_path(fs::path const & cache, std::string const & key) {
	return manifest_dir(cache) / (key + ".json");
}

fs::path sc::artifact_path(
	fs::path const & cache,
	std::string const & key,
	char const * const extension
) {
	return artifact_dir(cache) / (key + extension);
}

// -----------------------------------------------------------------------------
// -- manifest
// -----------------------------------------------------------------------------

std::string sc::manifest_serialize(Variant const & variant) {
	Variant const normalized = variant_normalized(variant);
	rapidjson::StringBuffer buffer;
	JsonWriter writer(buffer);
	writer.StartObject();
	writer.Key("entry");
	writer.String(normalized.entry.c_str());
	writer.Key("defines");
	writer.StartArray();
	for (std::string const & define : normalized.defines) {
		writer.String(define.c_str());
	}
	writer.EndArray();
	writer.Key("optimize");
	writer.Bool(normalized.optimize);
	writer.Key("deferred");
	writer.Bool(normalized.deferred);
	writer.EndObject();
	return buffer.GetString();
}

std::optional<sc::Variant> sc::manifest_parse(std::string const & json) {
	std::optional<rapidjson::Document> const document = json_parse(json);
	if (!document) { return std::nullopt; }
	rapidjson::Document const & d = *document;
	if (
		!d.HasMember("entry") || !d["entry"].IsString()
		|| !d.HasMember("defines") || !d["defines"].IsArray()
		|| !d.HasMember("optimize") || !d["optimize"].IsBool()
	) {
		return std::nullopt;
	}
	Variant variant {
		.entry = d["entry"].GetString(),
		.defines = {},
		.optimize = d["optimize"].GetBool(),
		// absent in manifests written before it existed
		.deferred = (
			d.HasMember("deferred") && d["deferred"].IsBool()
			&& d["deferred"].GetBool()
		),
	};
	for (auto const & define : d["defines"].GetArray()) {
		if (!define.IsString()) { return std::nullopt; }
		variant.defines.emplace_back(define.GetString());
	}
	return variant_normalized(variant);
}

// -----------------------------------------------------------------------------
// -- artifact
// -----------------------------------------------------------------------------

std::string sc::artifact_serialize(Artifact const & artifact) {
	rapidjson::StringBuffer buffer;
	JsonWriter writer(buffer);
	writer.StartObject();
	writer.Key("generation");
	writer.Uint64(artifact.generation);
	writer.Key("ok");
	writer.Bool(artifact.ok);
	writer.Key("error");
	writer.String(artifact.error.c_str());
	writer.Key("hasBinary");
	writer.Bool(artifact.hasBinary);
	writer.Key("slangMs");
	writer.Double(artifact.slangMs);
	writer.Key("vulkanMs");
	writer.Double(artifact.vulkanMs);
	if (artifact.statistics) {
		Statistics const & statistics = *artifact.statistics;
		writer.Key("statistics");
		writer.StartObject();
		writer.Key("registers");
		writer.Uint(statistics.registers);
		writer.Key("spillBytes");
		writer.Uint(statistics.spillBytes);
		writer.Key("sharedBytes");
		writer.Uint(statistics.sharedBytes);
		writer.Key("stackBytes");
		writer.Uint(statistics.stackBytes);
		writer.Key("isaBytes");
		writer.Uint(statistics.isaBytes);
		writer.Key("occupancyPercent");
		writer.Uint(statistics.occupancyPercent);
		writer.Key("occupancyWarps");
		writer.Uint(statistics.occupancyWarps);
		writer.Key("warpsPerSm");
		writer.Uint(statistics.warpsPerSm);
		writer.Key("extra");
		writer.String(statistics.extra.c_str());
		writer.EndObject();
	}
	writer.Key("dependencies");
	writer.StartArray();
	for (Dependency const & dependency : artifact.dependencies) {
		writer.StartObject();
		writer.Key("path");
		writer.String(dependency.path.c_str());
		writer.Key("hash");
		writer.String(hex64(dependency.hash).c_str());
		writer.EndObject();
	}
	writer.EndArray();
	writer.EndObject();
	return buffer.GetString();
}

static std::optional<sc::Statistics> statistics_parse(
	rapidjson::Document const & d
) {
	if (!d.HasMember("statistics") || !d["statistics"].IsObject()) {
		return std::nullopt;
	}
	auto const & object = d["statistics"];
	auto const fnU32 = [&](char const * const name) -> u32 {
		return (
			object.HasMember(name) && object[name].IsUint()
			? object[name].GetUint()
			: 0u
		);
	};
	return sc::Statistics {
		.registers = fnU32("registers"),
		.spillBytes = fnU32("spillBytes"),
		.sharedBytes = fnU32("sharedBytes"),
		.stackBytes = fnU32("stackBytes"),
		.isaBytes = fnU32("isaBytes"),
		.occupancyPercent = fnU32("occupancyPercent"),
		.occupancyWarps = fnU32("occupancyWarps"),
		.warpsPerSm = fnU32("warpsPerSm"),
		.extra = (
			object.HasMember("extra") && object["extra"].IsString()
			? object["extra"].GetString()
			: ""
		),
	};
}

std::optional<sc::Artifact> sc::artifact_parse(std::string const & json) {
	std::optional<rapidjson::Document> const document = json_parse(json);
	if (!document) { return std::nullopt; }
	rapidjson::Document const & d = *document;
	if (
		!d.HasMember("generation") || !d["generation"].IsUint64()
		|| !d.HasMember("ok") || !d["ok"].IsBool()
		|| !d.HasMember("error") || !d["error"].IsString()
		|| !d.HasMember("hasBinary") || !d["hasBinary"].IsBool()
		|| !d.HasMember("dependencies") || !d["dependencies"].IsArray()
	) {
		return std::nullopt;
	}
	auto const fnNumber = [&](char const * const name) -> f64 {
		return (
			d.HasMember(name) && d[name].IsNumber() ? d[name].GetDouble() : 0.0
		);
	};
	Artifact artifact {
		.generation = d["generation"].GetUint64(),
		.ok = d["ok"].GetBool(),
		.error = d["error"].GetString(),
		.dependencies = {},
		.hasBinary = d["hasBinary"].GetBool(),
		.slangMs = fnNumber("slangMs"),
		.vulkanMs = fnNumber("vulkanMs"),
		.statistics = statistics_parse(d),
	};
	for (auto const & dependency : d["dependencies"].GetArray()) {
		if (
			!dependency.IsObject()
			|| !dependency.HasMember("path") || !dependency["path"].IsString()
			|| !dependency.HasMember("hash") || !dependency["hash"].IsString()
		) {
			return std::nullopt;
		}
		std::optional<u64> const hash = (
			hex64_parse(dependency["hash"].GetString())
		);
		if (!hash) { return std::nullopt; }
		artifact.dependencies.emplace_back(Dependency {
			.path = dependency["path"].GetString(),
			.hash = *hash,
		});
	}
	return artifact;
}

std::optional<sc::Artifact> sc::artifact_read(
	fs::path const & cache,
	std::string const & key
) {
	std::optional<std::string> const json = (
		file_read(artifact_path(cache, key, ".json"))
	);
	if (!json) { return std::nullopt; }
	return artifact_parse(*json);
}

bool sc::artifact_fresh(Artifact const & artifact, fs::path const & root) {
	if (artifact.dependencies.empty()) { return false; }
	for (Dependency const & dependency : artifact.dependencies) {
		if (file_hash(root / dependency.path) != dependency.hash) {
			return false;
		}
	}
	return true;
}

// -----------------------------------------------------------------------------
// -- daemon state
// -----------------------------------------------------------------------------

std::string sc::heartbeat_serialize(Heartbeat const & heartbeat) {
	return (
		std::to_string(heartbeat.instance)
		+ " "
		+ std::to_string(heartbeat.unixMs)
	);
}

std::optional<sc::Heartbeat> sc::heartbeat_read(fs::path const & cache) {
	std::optional<std::string> const text = file_read(lock_path(cache));
	if (!text) { return std::nullopt; }
	unsigned long long instance = 0ull;
	long long unixMs = 0ll;
	if (sscanf(text->c_str(), "%llu %lld", &instance, &unixMs) != 2) {
		return std::nullopt;
	}
	return Heartbeat { .instance = (u64)instance, .unixMs = (i64)unixMs };
}

bool sc::daemon_alive(fs::path const & cache) {
	std::optional<Heartbeat> const heartbeat = heartbeat_read(cache);
	return (
		heartbeat.has_value()
		&& now_unix_ms() - heartbeat->unixMs < skHeartbeatStaleMs
	);
}

u64 sc::generation_read(fs::path const & cache) {
	std::optional<std::string> const text = file_read(generation_path(cache));
	if (!text) { return 0ull; }
	unsigned long long generation = 0ull;
	if (sscanf(text->c_str(), "%llu", &generation) != 1) { return 0ull; }
	return (u64)generation;
}
