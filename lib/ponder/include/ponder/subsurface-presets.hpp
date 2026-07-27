#pragma once

#include <srat/core-math.hpp>

namespace ponder {

// radiusFraction is relative to the mesh's own bounding-box diagonal, not an
// absolute distance -- see subsurfaceRadiusFromExtent below
struct SubsurfacePreset {
	char const * name;
	f32v3 color;
	f32 radiusFraction;
	f32v3 radiusScale;
	f32 anisotropy;
};

// clamped below skMaxSubsurfaceAnisotropy (0.95,
// util-material-openpbr-subsurface.glsl) with margin -- the phase sampler
// has a NaN edge case at exactly 1.0
inline constexpr SubsurfacePreset kSubsurfacePresets[] = {
	{
		.name = "skin",
		.color = { 0.72f, 0.45f, 0.36f },
		.radiusFraction = 0.015f,
		.radiusScale = { 1.0f, 0.4f, 0.25f },
		.anisotropy = 0.0f,
	},
	{
		.name = "wax",
		.color = { 0.9f, 0.85f, 0.7f },
		.radiusFraction = 0.05f,
		.radiusScale = { 1.0f, 0.9f, 0.75f },
		.anisotropy = 0.3f,
	},
	{
		.name = "marble",
		.color = { 0.95f, 0.93f, 0.88f },
		.radiusFraction = 0.08f,
		.radiusScale = { 1.0f, 0.95f, 0.9f },
		.anisotropy = 0.0f,
	},
	{
		.name = "jade",
		.color = { 0.55f, 0.85f, 0.6f },
		.radiusFraction = 0.04f,
		.radiusScale = { 0.6f, 1.0f, 0.7f },
		.anisotropy = 0.1f,
	},
	{
		.name = "milk",
		.color = { 0.98f, 0.97f, 0.93f },
		.radiusFraction = 0.1f,
		.radiusScale = { 1.0f, 1.0f, 0.95f },
		.anisotropy = 0.6f,
	},
};

inline f32 subsurfaceRadiusFromExtent(
	f32v3 const boundsMin, f32v3 const boundsMax, f32 const radiusFraction
) {
	f32 const diagonal = f32v3_length(boundsMax - boundsMin);
	return diagonal * radiusFraction;
}

} // namespace ponder
