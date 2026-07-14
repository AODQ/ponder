#pragma once

#include <string>
#include <vkof/vkof.hpp>

namespace ponder {

// represents an HDR environment map and its precomputed
// importance-sampling tables (pdf, conditional cdf, marginal cdf)
struct EnvironmentTables {
	vkof::Sampler radianceSampler {};
	vkof::Sampler tableSampler;
	vkof::Image radianceImage {};
	vkof::Image pdfImage;
	vkof::Image conditionalCdfImage;
	vkof::Image marginalCdfImage;
	u32 radianceHandle;
	u32 pdfHandle;
	u32 conditionalCdfHandle;
	u32 marginalCdfHandle;
	u32 width;
	u32 height;
};

EnvironmentTables environment_tables_create(const std::string & path);
void environment_tables_destroy(EnvironmentTables & tables);

} // namespace ponder
