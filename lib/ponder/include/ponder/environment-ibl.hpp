#pragma once

#include <ponder/environment-tables.hpp>

#include <vkof/vkof.hpp>

#include <vector>

namespace ponder {

struct EnvironmentIbl {
	static constexpr u32 skSpecularMipCount = 6u;
	static constexpr u32 skSpecularBaseWidth = 512u;
	static constexpr u32 skSpecularBaseHeight = 256u;
	static constexpr u32 skIrradianceWidth = 32u;
	static constexpr u32 skIrradianceHeight = 16u;

	vkof::Sampler sampler {};
	vkof::Image specularImage {};
	vkof::Image irradianceImage {};
	u32 specularHandle {};
	u32 irradianceHandle {};

	vkof::Pipeline specularPipeline {};
	vkof::Pipeline irradiancePipeline {};
};

[[nodiscard]] EnvironmentIbl environment_ibl_create();
void environment_ibl_destroy(EnvironmentIbl & ibl);

// vkof has no immediate-submit path, so this rides the caller's own
// render_graph_execute rather than a separate pre-loop submission
void environment_ibl_bake(
	EnvironmentIbl const & ibl,
	EnvironmentTables const & env,
	std::vector<vkof::RenderNode> & nodes
);

} // namespace ponder
