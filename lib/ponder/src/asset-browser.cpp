#include <ponder/asset-browser.hpp>

#include <imgui.h>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

#include <algorithm>
#include <cstdio>

std::vector<ponder::ModelEntry> ponder::scan_asset_paths(
	std::filesystem::path const & settingsPath,
	std::initializer_list<char const *> const keys,
	std::initializer_list<char const *> const extensions,
	bool const warnMissingKey
) {
	std::vector<ModelEntry> out;
	FILE * const file = fopen(settingsPath.string().c_str(), "rb");
	if (!file) {
		return out;
	}
	char buffer[16384];
	rapidjson::FileReadStream stream(file, buffer, sizeof(buffer));
	rapidjson::Document doc;
	doc.ParseStream(stream);
	fclose(file);
	if (doc.HasParseError() || !doc.IsObject()) {
		printf("failed to parse %s\n", settingsPath.string().c_str());
		return out;
	}
	auto pathsIt = doc.MemberEnd();
	for (char const * const key : keys) {
		pathsIt = doc.FindMember(key);
		if (pathsIt != doc.MemberEnd() && pathsIt->value.IsArray()) {
			break;
		}
	}
	if (pathsIt == doc.MemberEnd() || !pathsIt->value.IsArray()) {
		if (warnMissingKey) {
			printf(
				"%s has no \"%s\" array\n",
				settingsPath.string().c_str(), *keys.begin()
			);
		}
		return out;
	}
	for (auto const & entry : pathsIt->value.GetArray()) {
		if (!entry.IsString()) {
			continue;
		}
		std::filesystem::path dir(entry.GetString());
		if (dir.is_relative()) {
			dir = settingsPath.parent_path() / dir;
		}
		std::error_code ec;
		if (!std::filesystem::is_directory(dir, ec)) {
			printf(
				"\"%s\" entry is not a directory: %s\n",
				*keys.begin(), dir.string().c_str()
			);
			continue;
		}
		// sorted per-root rather than globally, so the imgui grouping below
		// gets contiguous, alphabetized runs within each root's collapsible
		// section instead of needing to re-sort at display time
		std::vector<std::filesystem::path> found;
		for (
			std::filesystem::recursive_directory_iterator it(
				dir,
				std::filesystem::directory_options::skip_permission_denied,
				ec
			);
			it != std::filesystem::recursive_directory_iterator();
			it.increment(ec)
		) {
			if (ec) {
				break;
			}
			if (!it->is_regular_file(ec)) {
				continue;
			}
			std::filesystem::path const & p = it->path();
			for (char const * const ext : extensions) {
				if (p.extension() == ext) {
					found.push_back(p);
					break;
				}
			}
		}
		std::sort(found.begin(), found.end());
		for (std::filesystem::path & p : found) {
			out.push_back({ dir, std::move(p) });
		}
	}
	return out;
}

ponder::AssetTagIndex ponder::asset_tag_index_load(
	std::filesystem::path const & indexPath
) {
	AssetTagIndex out;
	out.root = indexPath.parent_path().lexically_normal();
	FILE * const file = fopen(indexPath.string().c_str(), "rb");
	if (!file) {
		return out;
	}
	char buffer[16384];
	rapidjson::FileReadStream stream(file, buffer, sizeof(buffer));
	rapidjson::Document doc;
	doc.ParseStream(stream);
	fclose(file);
	if (doc.HasParseError() || !doc.IsObject()) {
		printf("failed to parse %s\n", indexPath.string().c_str());
		return out;
	}
	auto const tagsIt = doc.FindMember("tags");
	if (tagsIt != doc.MemberEnd() && tagsIt->value.IsObject()) {
		for (auto const & tag : tagsIt->value.GetObject()) {
			out.tagDescriptions[tag.name.GetString()] = (
				tag.value.IsString() ? tag.value.GetString() : ""
			);
		}
	}
	auto const assetsIt = doc.FindMember("assets");
	if (assetsIt != doc.MemberEnd() && assetsIt->value.IsObject()) {
		for (auto const & asset : assetsIt->value.GetObject()) {
			if (!asset.value.IsArray()) {
				continue;
			}
			std::vector<std::string> & tags = (
				out.assetTags[asset.name.GetString()]
			);
			for (auto const & tag : asset.value.GetArray()) {
				if (!tag.IsString()) {
					continue;
				}
				tags.emplace_back(tag.GetString());
				// tolerate tags missing from the "tags" table
				out.tagDescriptions.try_emplace(tag.GetString(), "");
			}
		}
	}
	return out;
}

std::vector<std::string> const * ponder::asset_tags_find(
	AssetTagIndex const & index, std::filesystem::path const & assetPath
) {
	std::filesystem::path const rel = (
		assetPath.lexically_normal().lexically_relative(index.root)
	);
	if (rel.empty() || *rel.begin() == "..") {
		return nullptr;
	}
	auto it = index.assetTags.find(rel.generic_string());
	if (it != index.assetTags.end()) {
		return &it->second;
	}
	for (
		std::filesystem::path dir = rel.parent_path();
		!dir.empty();
		dir = dir.parent_path()
	) {
		it = index.assetTags.find(dir.generic_string() + "/");
		if (it != index.assetTags.end()) {
			return &it->second;
		}
	}
	return nullptr;
}

bool ponder::asset_tag_filter_pass(
	AssetTagFilter const & filter, std::filesystem::path const & assetPath
) {
	if (!filter.index || filter.selected.empty()) {
		return true;
	}
	std::vector<std::string> const * const tags = (
		asset_tags_find(*filter.index, assetPath)
	);
	if (!tags) {
		return false;
	}
	usize hits = 0u;
	for (std::string const & tag : *tags) {
		hits += filter.selected.count(tag);
	}
	return filter.matchAny ? hits > 0u : hits == filter.selected.size();
}

void ponder::asset_tag_filter_draw(
	AssetTagFilter & filter, char const * const id
) {
	if (!filter.index || filter.index->tagDescriptions.empty()) {
		ImGui::TextDisabled("no asset tag index loaded");
		return;
	}
	ImGui::PushID(id);
	ImGui::Checkbox("match any", &filter.matchAny);
	ImGui::SameLine();
	if (ImGui::SmallButton("clear")) {
		filter.selected.clear();
	}
	ImGuiStyle const & style = ImGui::GetStyle();
	f32 const right = (
		ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x
	);
	bool first = true;
	for (auto const & [tag, description] : filter.index->tagDescriptions) {
		f32 const width = (
			ImGui::CalcTextSize(tag.c_str()).x + style.FramePadding.x * 2.0f
		);
		if (!first) {
			ImGui::SameLine();
			if (ImGui::GetCursorScreenPos().x + width > right) {
				ImGui::NewLine();
			}
		}
		first = false;
		bool const on = filter.selected.count(tag) > 0u;
		if (on) {
			ImGui::PushStyleColor(
				ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive)
			);
		}
		ImGui::PushID(tag.c_str());
		if (ImGui::SmallButton(tag.c_str())) {
			if (on) {
				filter.selected.erase(tag);
			} else {
				filter.selected.insert(tag);
			}
		}
		ImGui::PopID();
		if (on) {
			ImGui::PopStyleColor();
		}
		if (!description.empty()) {
			ImGui::SetItemTooltip("%s", description.c_str());
		}
	}
	ImGui::PopID();
}

std::string ponder::asset_list_draw(
	std::vector<ModelEntry> const & list,
	std::string const & currentPath,
	char const * const childId,
	f32 const height,
	AssetTagFilter const * const filter
) {
	std::string picked;
	ImGui::BeginChild(childId, ImVec2(0.0f, height), ImGuiChildFlags_None);
	std::filesystem::path const * openRoot = nullptr;
	bool rootOpen = false;
	for (ModelEntry const & entry : list) {
		if (!openRoot || *openRoot != entry.root) {
			openRoot = &entry.root;
			ImGui::PushID(entry.root.string().c_str());
			rootOpen = ImGui::CollapsingHeader(
				entry.root.string().c_str(),
				ImGuiTreeNodeFlags_DefaultOpen
			);
			ImGui::PopID();
		}
		if (!rootOpen || (filter && !asset_tag_filter_pass(*filter, entry.path))) {
			continue;
		}
		// parent-dir/filename: distinguishes the gltf/gltf-binary variants
		// sample-asset directories usually keep side by side under the same
		// model folder
		std::string const label = (
			entry.path.parent_path().filename().string()
			+ "/" + entry.path.filename().string()
		);
		bool const isCurrent = entry.path.string() == currentPath;
		// full path as id: labels repeat across roots (two libraries can
		// both have an "Avocado" folder)
		ImGui::PushID(entry.path.string().c_str());
		if (ImGui::Selectable(label.c_str(), isCurrent) && !isCurrent) {
			picked = entry.path.string();
		}
		if (filter && filter->index && ImGui::IsItemHovered()) {
			std::vector<std::string> const * const tags = (
				asset_tags_find(*filter->index, entry.path)
			);
			if (tags) {
				std::string joined;
				for (std::string const & tag : *tags) {
					joined += joined.empty() ? tag : ", " + tag;
				}
				ImGui::SetTooltip("%s", joined.c_str());
			}
		}
		ImGui::PopID();
	}
	ImGui::EndChild();
	return picked;
}
