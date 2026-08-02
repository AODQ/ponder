#include <ponder/material-media.hpp>

#include <imgui.h>

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <vector>

namespace {

// mirrors MaterialMediumRecord in shaders/medium/material-medium.slang
struct GpuMaterialMedium {
	u32 type;
	u32 phase;
	u32 priority;
	u32 pad0;
	f32v3 sigmaAbsorption;
	f32 anisotropy;
	f32v3 sigmaScattering;
	f32 draineAlpha;
	f32 dropletDiameter;
	f32 heightOrigin;
	f32 heightFalloff;
	u32 pad1;
};
static_assert(sizeof(GpuMaterialMedium) == 64u);
static_assert(offsetof(GpuMaterialMedium, sigmaAbsorption) == 16u);
static_assert(offsetof(GpuMaterialMedium, sigmaScattering) == 32u);
static_assert(offsetof(GpuMaterialMedium, dropletDiameter) == 48u);

} // namespace

static GpuMaterialMedium gpu_material_medium(
	ponder::MaterialMedium const & medium
) {
	return GpuMaterialMedium {
		.type = (u32)medium.type,
		.phase = (u32)medium.phase,
		.priority = medium.priority,
		.pad0 = 0u,
		.sigmaAbsorption = medium.absorption * medium.density,
		.anisotropy = medium.anisotropy,
		.sigmaScattering = medium.scattering * medium.density,
		.draineAlpha = medium.draineAlpha,
		.dropletDiameter = medium.dropletDiameter,
		.heightOrigin = medium.heightOrigin,
		.heightFalloff = medium.heightFalloff,
		.pad1 = 0u,
	};
}

void ponder::material_media_destroy(MaterialMedia & materialMedia) {
	if (materialMedia.tableBuffer.id) {
		vkof::buffer_destroy(materialMedia.tableBuffer);
	}
	if (materialMedia.recordsBuffer.id) {
		vkof::buffer_destroy(materialMedia.recordsBuffer);
	}
	materialMedia.tableBuffer = {};
	materialMedia.recordsBuffer = {};
	materialMedia.modelCount = 0u;
}

void ponder::material_media_upload(MaterialMedia & materialMedia) {
	// -- free in-flight rseources
	vkof::device_wait_idle();
	material_media_destroy(materialMedia);
	if (materialMedia.media.empty()) {
		return;
	}

	// -- count model and material count
	i32 modelCount = 0;
	std::map<i32, i32> materialCounts;
	for (auto const & [key, medium] : materialMedia.media) {
		modelCount = std::max(modelCount, key.first + 1);
		materialCounts[key.first] = (
			std::max(materialCounts[key.first], key.second + 1)
		);
	}

	// -- create the medium table with offsets into the records buffer
	std::vector<u32v2> table((usize)modelCount, u32v2 { 0u, 0u });
	std::vector<GpuMaterialMedium> records;
	for (auto const & [model, count] : materialCounts) {
		table[(usize)model] = u32v2 { (u32)records.size(), (u32)count };
		for (i32 material = 0; material < count; ++material) {
			auto const found = materialMedia.media.find({ model, material });
			records.emplace_back(
				found != materialMedia.media.end()
				? gpu_material_medium(found->second)
				: gpu_material_medium(MaterialMedium {})
			);
		}
	}

	// -- upload the table
	materialMedia.modelCount = (u32)modelCount;
	materialMedia.tableBuffer = vkof::buffer_create({
		.byteCount = table.size() * sizeof(u32v2),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	vkof::buffer_upload({
		.buffer = materialMedia.tableBuffer,
		.byteOffset = 0u,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(table.data()),
			(u32)(table.size() * sizeof(u32v2))
		),
	});

	// -- upload the records
	materialMedia.recordsBuffer = vkof::buffer_create({
		.byteCount = records.size() * sizeof(GpuMaterialMedium),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	vkof::buffer_upload({
		.buffer = materialMedia.recordsBuffer,
		.byteOffset = 0u,
		.data = (
			srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(records.data()),
				(u32)(records.size() * sizeof(GpuMaterialMedium))
			)
		),
	});
}

bool ponder::material_media_list_imgui(
	MaterialMedia const & materialMedia,
	std::vector<MaterialMediaEntry> const & entries,
	i32 & modelDrawIndex,
	i32 & materialIndex
) {
	bool picked = false;
	if (!ImGui::Begin("Scene materials")) {
		ImGui::End();
		return false;
	}
	if (entries.empty()) {
		ImGui::TextDisabled("no scene loaded");
	}

	// -- one tree node per model, one row per material
	usize i = 0u;
	while (i < entries.size()) {
		i32 const model = entries[i].modelDrawIndex;
		ImGui::PushID(model);
		bool const open = ImGui::TreeNodeEx(
			"##model",
			ImGuiTreeNodeFlags_DefaultOpen,
			"%s",
			entries[i].modelName.c_str()
		);
		for (; i < entries.size() && entries[i].modelDrawIndex == model; ++i) {
			if (!open) { continue; }
			MaterialMediaEntry const & entry = entries[i];

			// -- materials that bound a medium are tagged
			bool const isMedium = materialMedia.media.contains(
				{ entry.modelDrawIndex, entry.materialIndex }
			);
			ImGui::PushID(entry.materialIndex);
			bool const selected = (
				entry.modelDrawIndex == modelDrawIndex
				&& entry.materialIndex == materialIndex
			);
			std::string const label = (
				entry.materialName + (isMedium ? "  [medium]" : "")
			);
			if (ImGui::Selectable(label.c_str(), selected)) {
				modelDrawIndex = entry.modelDrawIndex;
				materialIndex = entry.materialIndex;
				picked = true;
			}
			ImGui::PopID();
		}
		if (open) {
			ImGui::TreePop();
		}
		ImGui::PopID();
	}
	ImGui::End();
	return picked;
}

bool ponder::material_media_imgui(
	MaterialMedia & materialMedia,
	i32 const modelDrawIndex,
	i32 const materialIndex
) {
	if (modelDrawIndex < 0 || materialIndex < 0) {
		return false;
	}
	auto const key = std::make_pair(modelDrawIndex, materialIndex);
	auto const found = materialMedia.media.find(key);
	MaterialMedium medium = (
		found != materialMedia.media.end() ? found->second : MaterialMedium {}
	);
	bool changed = false;

	// -- surface or medium
	ImGui::SeparatorText("medium");
	char const * const types[] = { "surface", "homogeneous", "height fog" };
	i32 type = (i32)medium.type;
	ImGui::TextUnformatted("the mesh is");
	ImGui::SetNextItemWidth(-FLT_MIN);
	changed |= ImGui::Combo(
		"##materialmediumtype", &type, types, IM_ARRAYSIZE(types)
	);
	medium.type = (WavefrontAmbientMediumType)type;

	if (medium.type != WavefrontAmbientMediumType::None) {
		// -- coefficients
		changed |= ImGui::DragFloat(
			"density##materialmedium",
			&medium.density,
			0.01f,
			0.0f,
			1000.0f,
			"%.3f"
		);
		ImGui::TextUnformatted("scattering (σs per meter)");
		ImGui::SetNextItemWidth(-FLT_MIN);
		changed |= ImGui::ColorEdit3(
			"##materialmediumscattering",
			&medium.scattering.x,
			ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR
		);
		ImGui::TextUnformatted("absorption (σa per meter)");
		ImGui::SetNextItemWidth(-FLT_MIN);
		changed |= ImGui::ColorEdit3(
			"##materialmediumabsorption",
			&medium.absorption.x,
			ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR
		);

		// -- priority where media overlap
		i32 priority = (i32)medium.priority;
		changed |= ImGui::SliderInt(
			"priority##materialmedium", &priority, 0, 16
		);
		medium.priority = (u32)std::max(priority, 0);

		// -- phase function
		char const * const phases[] = {
			"henyey-greenstein", "draine", "fog (droplets)",
		};
		i32 phase = (i32)medium.phase;
		ImGui::TextUnformatted("phase");
		ImGui::SetNextItemWidth(-FLT_MIN);
		changed |= (
			ImGui::Combo(
				"##materialmediumphase", &phase, phases, IM_ARRAYSIZE(phases)
			)
		);
		medium.phase = (WavefrontAmbientMediumPhase)phase;
		if (medium.phase != WavefrontAmbientMediumPhase::Fog) {
			changed |= (
				ImGui::SliderFloat(
					"g##materialmedium", &medium.anisotropy, -0.99f, 0.99f
				)
			);
		}
		if (medium.phase == WavefrontAmbientMediumPhase::Draine) {
			changed |= (
				ImGui::SliderFloat(
					"α##materialmedium", &medium.draineAlpha, 0.0f, 50.0f
				)
			);
		}
		if (medium.phase == WavefrontAmbientMediumPhase::Fog) {
			ImGui::TextUnformatted("droplet diameter (µm)");
			ImGui::SetNextItemWidth(-FLT_MIN);
			changed |= (
				ImGui::SliderFloat(
					"##materialmediumdroplet", &medium.dropletDiameter, 5.0f, 50.0f
				)
			);
		}

		// -- height fog
		if (medium.type == WavefrontAmbientMediumType::HeightFog) {
			changed |= (
				ImGui::DragFloat(
					"origin##materialmedium", &medium.heightOrigin, 0.05f
				)
			);
			changed |= (
				ImGui::DragFloat(
					"falloff##materialmedium",
					&medium.heightFalloff,
					0.05f,
					0.01f,
					1000.0f
				)
			);
		}
	}

	// -- store and upload
	// a surface needs no record
	if (!changed) {
		return false;
	}
	if (medium.type == WavefrontAmbientMediumType::None) {
		materialMedia.media.erase(key);
	} else {
		materialMedia.media[key] = medium;
	}
	material_media_upload(materialMedia);
	return true;
}

u64 ponder::material_media_table_va(MaterialMedia const & materialMedia) {
	return (
		materialMedia.tableBuffer.id
		? vkof::buffer_virtual_address(materialMedia.tableBuffer)
		: 0u
	);
}

u64 ponder::material_media_records_va(MaterialMedia const & materialMedia) {
	return (
		materialMedia.recordsBuffer.id
		? vkof::buffer_virtual_address(materialMedia.recordsBuffer)
		: 0u
	);
}
