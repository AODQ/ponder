#pragma once

#include <ponder/wavefront.hpp>

#include <vkof/vkof.hpp>

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ponder {

struct MaterialMedium {
	WavefrontAmbientMediumType type { WavefrontAmbientMediumType::None };
	f32v3 scattering { 1.0f, 1.0f, 1.0f };
	f32v3 absorption { 0.0f, 0.0f, 0.0f };
	f32 density { 1.0f };
	WavefrontAmbientMediumPhase phase {
		WavefrontAmbientMediumPhase::HenyeyGreenstein
	};
	f32 anisotropy { 0.0f };
	f32 draineAlpha { 1.0f };
	f32 dropletDiameter { 10.0f };
	f32 heightOrigin { 0.0f };
	f32 heightFalloff { 5.0f };
	u32 priority { 1u };
};

struct MaterialMedia {
	std::map<std::pair<i32, i32>, MaterialMedium> media;
	vkof::Buffer tableBuffer {};
	vkof::Buffer recordsBuffer {};
	u32 modelCount {};
};

struct MaterialMediaEntry {
	i32 modelDrawIndex;
	i32 materialIndex;
	std::string modelName;
	std::string materialName;
};

void material_media_destroy(MaterialMedia & materialMedia);

[[nodiscard]] bool material_media_list_imgui(
	MaterialMedia const & materialMedia,
	std::vector<MaterialMediaEntry> const & entries,
	i32 & modelDrawIndex,
	i32 & materialIndex
);

void material_media_upload(MaterialMedia & materialMedia);

[[nodiscard]] bool material_media_imgui(
	MaterialMedia & materialMedia,
	i32 const modelDrawIndex,
	i32 const materialIndex
);

[[nodiscard]] u64 material_media_table_va(
	MaterialMedia const & materialMedia
);
[[nodiscard]] u64 material_media_records_va(
	MaterialMedia const & materialMedia
);

} // namespace ponder
