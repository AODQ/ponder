#pragma once

#include <srat/core-types.hpp>

#include <filesystem>
#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
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

// assets/index.json: { "tags": { tag: description }, "assets": { path: [tag] } }
// paths are relative to the index file's directory; a key ending in '/'
// tags everything under that directory
struct AssetTagIndex {
	std::filesystem::path root;
	std::map<std::string, std::string> tagDescriptions;
	std::unordered_map<std::string, std::vector<std::string>> assetTags;
};

AssetTagIndex asset_tag_index_load(std::filesystem::path const & indexPath);

// nullptr when the asset is outside the index root or untagged
std::vector<std::string> const * asset_tags_find(
	AssetTagIndex const & index, std::filesystem::path const & assetPath
);

struct AssetTagFilter {
	AssetTagIndex const * index = nullptr;
	std::set<std::string> selected;
	bool matchAny = false;
};

bool asset_tag_filter_pass(
	AssetTagFilter const & filter, std::filesystem::path const & assetPath
);

// toggleable tag chips wrapped to the window width
void asset_tag_filter_draw(AssetTagFilter & filter, char const * const id);

// draws one grouped asset list (a collapsing header per configured root
// directory, parent-dir/filename labels) inside a fixed-height child.
// returns the clicked path, empty when nothing was picked this frame
std::string asset_list_draw(
	std::vector<ModelEntry> const & list,
	std::string const & currentPath,
	char const * const childId,
	f32 const height,
	AssetTagFilter const * const filter = nullptr
);

} // namespace ponder
