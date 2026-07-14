#pragma once

#include <vkof/vkof.hpp>

namespace ponder {

// zeltner, burley & chiang 2022 "practical multiple-scattering sheen using
// linearly transformed cosines" LTC parameter fit, 32x32 (roughness x mu),
// (a, b, R) packed rgb (a = 0). consumed by utilZeltnerFuzzLookup in
// util-material-openpbr-fuzz.glsl
struct ZeltnerTables {
	vkof::Sampler sampler {};
	vkof::Image zeltnerLtcParamImage {};
	u32 zeltnerLtcParamHandle {};
};

[[nodiscard]] ZeltnerTables zeltner_tables_create();
void zeltner_tables_destroy(ZeltnerTables & tables);

} // namespace ponder
