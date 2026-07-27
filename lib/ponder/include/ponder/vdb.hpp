#pragma once

#include <vkof/vkof.hpp>

#include <string>
#include <vector>

namespace ponder {

struct Vdb {
	vkof::Buffer gpuBuffer;
	// small descriptor pointed to by GpuGlobalExtended.vdb: the grid blob's
	// address plus the density maximum; layout mirrors VdbHandle in
	// shaders/util-vdb.glsl
	vkof::Buffer handleBuffer;
	// the serialized nanovdb grid, byte-identical to gpuBuffer's contents;
	// kept for cpu-side reads (bbox/stats queries, gpu crosscheck tests)
	// through nanovdb accessors
	std::vector<u8> blob;
	// grid-wide density maximum from the nanovdb stats (computed during
	// createNanoGrid); the delta/ratio-tracking majorant is this times the
	// integrator's sigma scale, and it must stay an upper bound -- an
	// underestimate biases the render rather than adding noise
	f32 densityMax = 0.0f;
};

void vdb_initialize();

// names of every FloatGrid in the file, in file order; used to enumerate
// which grids a caller can pass to vdb_load as gridName (a "blob")
[[nodiscard]] std::vector<std::string> vdb_grid_names(
	char const * const path
);

Vdb vdb_load(char const * const path, char const * const gridName = "density");

// grid's own world-space bounding box (nanovdb's worldBBox, no instance
// transform applied -- callers compose that themselves). false (out params
// untouched) if vdb has no loaded blob
[[nodiscard]] bool vdb_world_bounds(
	Vdb const & vdb, f32v3 & outMin, f32v3 & outMax
);

}
