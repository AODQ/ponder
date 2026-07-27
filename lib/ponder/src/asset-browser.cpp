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

std::string ponder::asset_list_draw(
	std::vector<ModelEntry> const & list,
	std::string const & currentPath,
	char const * const childId,
	f32 const height
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
		if (!rootOpen) {
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
		ImGui::PopID();
	}
	ImGui::EndChild();
	return picked;
}
