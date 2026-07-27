#pragma once

#include <string>
#include <vkof/vkof.hpp>

namespace ponder {

// represents an HDR environment map and its precomputed importance-sampling
// tables: pdf (per-texel solid-angle density, for MIS/direct-probe lookups)
// and an alias table (Vose's method -- O(1) sampling of the same
// distribution, replacing what used to be a marginal+conditional cdf
// binary search)
struct EnvironmentTables {
	vkof::Sampler radianceSampler {};
	vkof::Sampler tableSampler;
	vkof::Image radianceImage {};
	vkof::Image pdfImage;
	// alias table, same WxH as pdfImage: aliasProbImage is the per-texel
	// probability threshold, aliasIndexImage the redirect index (stored as
	// float -- no r32ui/r32g32 format available, and an index fits exactly
	// in f32 up to 2^24 texels)
	vkof::Image aliasProbImage;
	vkof::Image aliasIndexImage;
	u32 radianceHandle;
	u32 pdfHandle;
	u32 aliasProbHandle;
	u32 aliasIndexHandle;
	u32 width;
	u32 height;
	u32 radianceMipCount;
};

EnvironmentTables environment_tables_create(const std::string & path);
void environment_tables_destroy(EnvironmentTables & tables);

} // namespace ponder
