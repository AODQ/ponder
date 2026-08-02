#include <ponder/material-inspector.hpp>
#include <ponder/subsurface-presets.hpp>

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// -- private api
// -----------------------------------------------------------------------------

// material inspector popup: lets a texture slot be detached ("[none]") or
// swapped for any other texture already used elsewhere on this material --
// not a full asset browser, just cross-wiring between handles the material
// already references, which is all a gltf-sourced material ever has loaded
static bool fnMaterialTexturePicker(
	u32 & textureHandle,
	std::vector<std::string> const & textureNames,
	std::vector<u32> const & textureHandles
) {
	std::string texName = "[none]";
	{
		auto const it = (
			std::find(textureHandles.begin(), textureHandles.end(), textureHandle)
		);
		if (it != textureHandles.end()) {
			size_t const idx = std::distance(textureHandles.begin(), it);
			if (idx < textureNames.size()) {
				texName = textureNames[idx];
			}
		}
	}

	bool changed = false;
	if (ImGui::BeginCombo("##texture", texName.c_str())) {
		if (ImGui::Selectable("[none]", textureHandle == 0u)) {
			if (textureHandle != 0u) {
				textureHandle = 0u;
				changed = true;
			}
		}
		for (size_t i = 0u; i < textureNames.size(); ++i) {
			bool const selected = (textureHandles[i] == textureHandle);
			if (ImGui::Selectable(textureNames[i].c_str(), selected) && !selected) {
				textureHandle = textureHandles[i];
				changed = true;
			}
			if (selected) {
				ImGui::SetItemDefaultFocus();
			}
		}
		ImGui::EndCombo();
	}
	return changed;
}

// minVal/maxVal are unused here -- they exist so the
// MOR_MATERIAL_ALL_PARAMS_* x-macros (mor-shared.h) can pass every field's
// (lo, hi) uniformly and overload resolution alone picks this or the scalar
// overload below
static bool fnDrawMaterialSlot(
	GpuMorMaterialComponent3 & slot,
	char const * const label,
	std::vector<std::string> const & textureNames,
	std::vector<u32> const & textureHandles,
	[[maybe_unused]] f32 const minVal = 0.0f,
	[[maybe_unused]] f32 const maxVal = 1.0f
) {
	bool changed = false;
	ImGui::PushID(label);
	ImGui::Text("%s", label);
	changed |= ImGui::ColorEdit3("##value", &slot.rgb.x);
	changed |= (
		fnMaterialTexturePicker(slot.texture, textureNames, textureHandles)
	);
	ImGui::PopID();
	return changed;
}

static bool fnDrawMaterialSlot(
	GpuMorMaterialComponent1 & slot,
	char const * const label,
	std::vector<std::string> const & textureNames,
	std::vector<u32> const & textureHandles,
	f32 const minVal,
	f32 const maxVal
) {
	bool changed = false;
	ImGui::PushID(label);
	ImGui::Text("%s", label);
	changed |= ImGui::SliderFloat("##value", &slot.r, minVal, maxVal);
	ImGui::SameLine();
	// a wide range (e.g. emissionLuminance's 0-10000) makes the slider
	// above too coarse for fine adjustment; this drag is mouse-delta
	// relative instead of position-mapped, proportional to the current
	// value so it stays usable at any scale
	ImGui::SetNextItemWidth(24.0f);
	f32 const dragSpeed = std::max(std::fabs(slot.r) * 0.01f, 0.0001f);
	changed |= (
		ImGui::DragFloat("##drag", &slot.r, dragSpeed, minVal, maxVal, "")
	);
	changed |= (
		fnMaterialTexturePicker(slot.texture, textureNames, textureHandles)
	);
	if (slot.texture != 0u) {
		i32 const prevSwizzle = slot.swizzle;
		ImGui::RadioButton("R", &slot.swizzle, 0);
		ImGui::SameLine();
		ImGui::RadioButton("G", &slot.swizzle, 1);
		ImGui::SameLine();
		ImGui::RadioButton("B", &slot.swizzle, 2);
		ImGui::SameLine();
		ImGui::RadioButton("A", &slot.swizzle, 3);
		changed |= (slot.swizzle != prevSwizzle);
	}
	ImGui::PopID();
	return changed;
}

// every texture handle the material's slots reference, deduplicated and
// named after the slot that first uses it -- built from the original,
// unedited material so slot names stay meaningful even after edits
// reassign a texture to a different slot
static void fnMaterialTextureList(
	std::vector<std::string> & outNames,
	std::vector<u32> & outHandles,
	GpuMorMaterial const & origMat
) {
	auto const fnAddSlot = [&](char const * const name, u32 const handle) {
		if (handle == 0u) {
			return;
		}
		auto const it = (
			std::find(outHandles.begin(), outHandles.end(), handle)
		);
		if (it != outHandles.end()) {
			return;
		}
		outNames.emplace_back(name);
		outHandles.emplace_back(handle);
	};
#define X(Type, field, lo, hi) fnAddSlot(#field, origMat.field.texture);
	MOR_MATERIAL_ALL_PARAMS(X)
#undef X
}

// one collapsible section per OpenPBR layer, only shown once that layer's
// weight is nonzero (or texture-driven). every scalar/rgb parameter comes
// from mor-shared.h's MOR_MATERIAL_ALL_PARAMS_* x-macros, so a new material
// parameter shows up automatically
static bool fnImguiMaterial(
	GpuMorMaterial & p,
	GpuMorMaterial const & origMat,
	f32v3 const boundsMin,
	f32v3 const boundsMax
) {
	std::vector<std::string> textureNames;
	std::vector<u32> textureHandles;
	fnMaterialTextureList(textureNames, textureHandles, origMat);

	bool changed = false;
	ImGui::SeparatorText("weights");
#define X(Type, field, lo, hi) \
	changed |= ( \
		fnDrawMaterialSlot( \
			p.field, #field, textureNames, textureHandles, lo, hi \
		) \
	);
	MOR_MATERIAL_ALL_PARAMS_WEIGHTS(X)
#undef X
	if (p.baseWeight.r > 0.0f || p.baseWeight.texture != 0u) {
		ImGui::SeparatorText("base");
#define X(Type, field, lo, hi) \
		changed |= ( \
			fnDrawMaterialSlot( \
				p.field, #field, textureNames, textureHandles, lo, hi \
			) \
		);
		MOR_MATERIAL_ALL_PARAMS_BASE(X)
#undef X
	}
	if (p.specularWeight.r > 0.0f || p.specularWeight.texture != 0u) {
		ImGui::SeparatorText("specular");
#define X(Type, field, lo, hi) \
		changed |= ( \
			fnDrawMaterialSlot( \
				p.field, #field, textureNames, textureHandles, lo, hi \
			) \
		);
		MOR_MATERIAL_ALL_PARAMS_SPECULAR(X)
#undef X
	}
	if (p.transmissionWeight.r > 0.0f || p.transmissionWeight.texture != 0u) {
		ImGui::SeparatorText("transmission");
#define X(Type, field, lo, hi) \
		changed |= ( \
			fnDrawMaterialSlot( \
				p.field, #field, textureNames, textureHandles, lo, hi \
			) \
		);
		MOR_MATERIAL_ALL_PARAMS_TRANSMISSION(X)
#undef X
	}
	// preset picker always visible (not gated on subsurfaceWeight > 0) so
	// turning subsurface on for a material starts from a plausible look
	// instead of the flat global default
	ImGui::SeparatorText("subsurface");
	if (ImGui::BeginCombo("apply preset", "[choose a starting point]")) {
		for (ponder::SubsurfacePreset const & preset : ponder::kSubsurfacePresets) {
			if (ImGui::Selectable(preset.name)) {
				p.subsurfaceWeight.r = std::max(p.subsurfaceWeight.r, 1.0f);
				p.subsurfaceColor.rgb = preset.color;
				p.subsurfaceRadius.r = (
					ponder::subsurfaceRadiusFromExtent(
						boundsMin, boundsMax, preset.radiusFraction
					)
				);
				p.subsurfaceRadiusScale.rgb = preset.radiusScale;
				p.subsurfaceScatterAnisotropy.r = preset.anisotropy;
				changed = true;
			}
		}
		ImGui::EndCombo();
	}
	if (p.subsurfaceWeight.r > 0.0f || p.subsurfaceWeight.texture != 0u) {
#define X(Type, field, lo, hi) \
		changed |= ( \
			fnDrawMaterialSlot( \
				p.field, #field, textureNames, textureHandles, lo, hi \
			) \
		);
		MOR_MATERIAL_ALL_PARAMS_SUBSURFACE(X)
#undef X
	}
	if (p.coatWeight.r > 0.0f || p.coatWeight.texture != 0u) {
		ImGui::SeparatorText("coat");
#define X(Type, field, lo, hi) \
		changed |= ( \
			fnDrawMaterialSlot( \
				p.field, #field, textureNames, textureHandles, lo, hi \
			) \
		);
		MOR_MATERIAL_ALL_PARAMS_COAT(X)
#undef X
	}
	if (p.fuzzWeight.r > 0.0f || p.fuzzWeight.texture != 0u) {
		ImGui::SeparatorText("fuzz");
#define X(Type, field, lo, hi) \
		changed |= ( \
			fnDrawMaterialSlot( \
				p.field, #field, textureNames, textureHandles, lo, hi \
			) \
		);
		MOR_MATERIAL_ALL_PARAMS_FUZZ(X)
#undef X
	}
	if (p.emissionLuminance.r > 0.0f || p.emissionLuminance.texture != 0u) {
		ImGui::SeparatorText("emission");
#define X(Type, field, lo, hi) \
		changed |= ( \
			fnDrawMaterialSlot( \
				p.field, #field, textureNames, textureHandles, lo, hi \
			) \
		);
		MOR_MATERIAL_ALL_PARAMS_EMISSION(X)
#undef X
	}
	if (p.thinFilmWeight.r > 0.0f || p.thinFilmWeight.texture != 0u) {
		ImGui::SeparatorText("thin film");
#define X(Type, field, lo, hi) \
		changed |= ( \
			fnDrawMaterialSlot( \
				p.field, #field, textureNames, textureHandles, lo, hi \
			) \
		);
		MOR_MATERIAL_ALL_PARAMS_THIN_FILM(X)
#undef X
	}
	ImGui::SeparatorText("geometry");
#define X(Type, field, lo, hi) \
	changed |= ( \
		fnDrawMaterialSlot( \
			p.field, #field, textureNames, textureHandles, lo, hi \
		) \
	);
	MOR_MATERIAL_ALL_PARAMS_GEOMETRY(X)
#undef X
	ImGui::Text("alphaCutoff");
	changed |= ImGui::SliderFloat("##acutoff", &p.alphaCutoff, 0.0f, 1.0f);
	// (TODO REVIEW)
	ImGui::Text("alphaMode");
	{
		char const * const modeNames[] = { "opaque", "mask", "blend" };
		int mode = (int)p.alphaMode;
		if (ImGui::Combo("##amode", &mode, modeNames, 3)) {
			p.alphaMode = (u32)mode;
			changed = true;
		}
	}
	// (TODO REVIEW)
	return changed;
}

// -----------------------------------------------------------------------------
// -- public api
// -----------------------------------------------------------------------------

bool ponder::material_inspector_draw(
	mor::Scene const & scene,
	mor::GpuMaterials const & materialsOverride,
	i32 const materialIndex
) {
	if (
		materialIndex < 0
		|| (u32)materialIndex >= mor::scene_material_count(scene)
	) {
		ImGui::TextDisabled("no material at that pixel");
		return false;
	}
	std::string const name = (
		mor::scene_material_name(scene, (u32)materialIndex)
	);
	ImGui::Text("#%d %s", materialIndex, name.c_str());
	f32v3 boundsMin, boundsMax;
	mor::scene_bounds(scene, boundsMin, boundsMax);
	{
		f32 const extentX = boundsMax.x - boundsMin.x;
		f32 const extentY = boundsMax.y - boundsMin.y;
		f32 const extentZ = boundsMax.z - boundsMin.z;
		ImGui::Text(
			"scene bounds: (%.4f, %.4f, %.4f) to (%.4f, %.4f, %.4f)",
			boundsMin.x, boundsMin.y, boundsMin.z,
			boundsMax.x, boundsMax.y, boundsMax.z
		);
		ImGui::Text(
			"extent: %.4f x %.4f x %.4f (glTF units are in meters)",
			extentX, extentY, extentZ
		);
	}
	ImGui::Separator();
	GpuMorMaterial mat = (
		mor::scene_gpu_materials_get(materialsOverride, (u32)materialIndex)
	);
	GpuMorMaterial const origMat = (
		mor::scene_material_get(scene, (u32)materialIndex)
	);
	bool const changed = fnImguiMaterial(mat, origMat, boundsMin, boundsMax);
	if (changed) {
		mor::scene_material_override_scalars(
			materialsOverride, (u32)materialIndex, mat
		);
	}
	return changed;
}
