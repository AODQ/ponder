#pragma once

#include <srat/core-types.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vkof::shader_cache
{
	struct Variant {
		std::string entry;
		std::vector<std::string> defines;
		bool optimize;
		// build order only, not part of the key
		bool deferred { false };
	};

	struct Dependency {
		std::string path;
		u64 hash;
	};

	struct Statistics {
		u32 registers;
		u32 spillBytes;
		u32 sharedBytes;
		u32 stackBytes;
		u32 isaBytes;
		u32 occupancyPercent;
		u32 occupancyWarps;
		u32 warpsPerSm;
		std::string extra;
	};

	struct Artifact {
		u64 generation;
		bool ok;
		std::string error;
		std::vector<Dependency> dependencies;
		bool hasBinary;
		f64 slangMs;
		f64 vulkanMs;
		std::optional<Statistics> statistics;
	};

	struct Heartbeat {
		u64 instance;
		i64 unixMs;
	};

	std::string key(Variant const & variant);
	Variant variant_normalized(Variant variant);

	u64 bytes_hash(std::string_view bytes);
	u64 file_hash(std::filesystem::path const & path);
	std::optional<std::string> file_read(std::filesystem::path const & path);
	bool file_write_atomic(
		std::filesystem::path const & path,
		std::string_view bytes
	);
	i64 now_unix_ms();

	std::filesystem::path lock_path(std::filesystem::path const & cache);
	std::filesystem::path generation_path(std::filesystem::path const & cache);
	std::filesystem::path manifest_dir(std::filesystem::path const & cache);
	std::filesystem::path artifact_dir(std::filesystem::path const & cache);
	std::filesystem::path request_dir(std::filesystem::path const & cache);
	std::filesystem::path manifest_path(
		std::filesystem::path const & cache,
		std::string const & key
	);
	std::filesystem::path artifact_path(
		std::filesystem::path const & cache,
		std::string const & key,
		char const * extension
	);

	std::string manifest_serialize(Variant const & variant);
	std::optional<Variant> manifest_parse(std::string const & json);

	std::string artifact_serialize(Artifact const & artifact);
	std::optional<Artifact> artifact_parse(std::string const & json);
	std::optional<Artifact> artifact_read(
		std::filesystem::path const & cache,
		std::string const & key
	);
	bool artifact_fresh(
		Artifact const & artifact,
		std::filesystem::path const & root
	);

	std::string heartbeat_serialize(Heartbeat const & heartbeat);
	std::optional<Heartbeat> heartbeat_read(std::filesystem::path const & cache);
	bool daemon_alive(std::filesystem::path const & cache);
	u64 generation_read(std::filesystem::path const & cache);

	inline constexpr i64 skHeartbeatPeriodMs = 1000;
	inline constexpr i64 skHeartbeatStaleMs = 3000;
}
