#pragma once

#include <srat/core-types.hpp>

#include <filesystem>
#include <initializer_list>
#include <string>
#include <vector>

namespace ponder {

struct ModelEntry {
	std::filesystem::path root;
	std::filesystem::path path;
};

// settings.json at the repo root selects where assets come from:
//   { "modelPaths": ["assets/models", "/absolute/path/too"] }
// each listed directory (relative paths resolve against the settings
// file's own directory) is searched recursively for files matching one of
// `extensions`. the first key in `keys` whose value is an array wins --
// multiple keys only exist to accept spelling variants
// (environmentMapPaths / environmentMaps)
std::vector<ModelEntry> scan_asset_paths(
	std::filesystem::path const & settingsPath,
	std::initializer_list<char const *> const keys,
	std::initializer_list<char const *> const extensions,
	bool const warnMissingKey
);

// draws one grouped asset list (a collapsing header per configured root
// directory, parent-dir/filename labels) inside a fixed-height child.
// returns the clicked path, empty when nothing was picked this frame
std::string asset_list_draw(
	std::vector<ModelEntry> const & list,
	std::string const & currentPath,
	char const * const childId,
	f32 const height
);

} // namespace ponder
