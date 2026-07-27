#pragma once

#include <vkof/vkof.hpp>

namespace ponder {

// kulla-conty ggx directional albedo table, 16x16 (roughness x mu), plus
// its 16x1 hemispherical-average companion. reference/verification path
// alongside the analytic materialx rational fit (utilMicrofacetGgxDirectionalAlbedo)
struct EnergyTables {
	vkof::Sampler sampler {};
	vkof::Image kullaContyEnergyImage {};
	vkof::Image kullaContyEnergyAvgImage {};
	u32 kullaContyEnergyHandle {};
	u32 kullaContyEnergyAvgHandle {};
};

[[nodiscard]] EnergyTables energy_tables_create();
void energy_tables_destroy(EnergyTables const & tables);

} // namespace ponder
