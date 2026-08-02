#pragma once

#include <vkof/shader-cache.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace shaderd
{
	using HashSnapshot = std::unordered_map<std::string, u64>;

	struct CompileResult {
		bool ok;
		std::string diagnostics;
		std::string spirv;
		std::vector<vkof::shader_cache::Dependency> dependencies;
		u32 modulesReused;
		u32 modulesCompiled;
	};

	struct ModuleCache;
	struct ModuleCacheDeleter {
		void operator()(ModuleCache * cache) const;
	};
	using ModuleCachePtr = std::unique_ptr<ModuleCache, ModuleCacheDeleter>;

	ModuleCachePtr module_cache_create();

	std::string dependency_path(
		std::filesystem::path const & root,
		std::filesystem::path const & path
	);

	CompileResult compile(
		std::filesystem::path const & root,
		vkof::shader_cache::Variant const & variant,
		HashSnapshot const & snapshot,
		ModuleCache & cache
	);

	std::string slang_version();

	// releases pooled slang global sessions beyond keep; returns how many
	u32 global_session_pool_trim(u32 keep);
}
