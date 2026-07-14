#pragma once

// configuration for the tiled rasterizer

#define SRAT_DEBUG() SRAT_DEBUG_IMPL

#include <srat/core-types.hpp>

#define Let auto const &

#define defBinUseIndexCaching true

namespace config {
	static constexpr u32 skTileSize = 64;
	static size_t constexpr skSimdLaneWidth = 8;
}

//NOLINTBEGIN(cppcoreguidelines-macro-usage)

u64 & srat_tile_size();
bool & srat_temp_opt();
bool & srat_sequential_writes();
bool & srat_rasterize_parallel();
bool & srat_vertex_parallel();

// just a temp hack to change shader mode at runtime
enum struct ShaderMode {
	MaterialUnlit,
};

// -----------------------------------------------------------------------------
// -- tracy
// -----------------------------------------------------------------------------

#if defined(TRACY_ENABLE)
#define SRAT_TRACY_ENABLE() 1
#else
#define SRAT_TRACY_ENABLE() 0
#endif

#if SRAT_TRACY_ENABLE()
// disable all warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <tracy/Tracy.hpp>
#pragma GCC diagnostic pop
#define TracyZoneScoped ZoneScoped
#define TracyFrameMark FrameMark
#else
#define TracyZoneScoped
#define TracyFrameMark
#endif

//NOLINTEND(cppcoreguidelines-macro-usage)
