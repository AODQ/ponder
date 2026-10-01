// nanovdb loader + gpu upload
// might be its own lib in future but for now lives in ponder

#include <ponder/vdb.hpp>

#include <vkof/vkof.hpp>

#include <openvdb/openvdb.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>

// -----------------------------------------------------------------------------
// -- vdb_initialize
// -----------------------------------------------------------------------------

void ponder::vdb_initialize() {
	openvdb::initialize();
}

bool ponder::vdb_world_bounds(
	Vdb const & vdb, f32v3 & outMin, f32v3 & outMax
) {
	if (vdb.blob.empty()) { return false; }
	auto const * const grid = (
		reinterpret_cast<nanovdb::FloatGrid const *>(vdb.blob.data())
	);
	nanovdb::Vec3dBBox const bbox = grid->worldBBox();
	outMin = { (f32)bbox.min()[0], (f32)bbox.min()[1], (f32)bbox.min()[2] };
	outMax = { (f32)bbox.max()[0], (f32)bbox.max()[1], (f32)bbox.max()[2] };
	return true;
}

std::vector<std::string> ponder::vdb_grid_names(char const * const path) {
	std::vector<std::string> names;
	openvdb::io::File file(path);
	file.open();
	for (auto const & grid : *file.getGrids()) {
		names.emplace_back(grid->getName());
	}
	file.close();
	return names;
}

// -----------------------------------------------------------------------------
// -- vdb_load
// -----------------------------------------------------------------------------

ponder::Vdb ponder::vdb_load(
	char const * const path, char const * const gridName
) {
	Vdb vdb {};
	openvdb::io::File file(path);
	file.open();
	for (auto const & grid : *file.getGrids()) {
		if (grid->getName() != gridName) { continue; }
		auto const floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
		if (!floatGrid) { continue; }
		auto const nanoHandle = nanovdb::tools::createNanoGrid(*floatGrid);
		u8 const * const blobBegin = (
			reinterpret_cast<u8 const *>(nanoHandle.data())
		);
		vdb.blob.assign(blobBegin, blobBegin + nanoHandle.bufferSize());
		vdb.gpuBuffer = vkof::buffer_create({
			.byteCount = (u64)vdb.blob.size(),
			.memory = vkof::BufferMemory::DeviceOnly,
		});
		vkof::buffer_upload({
			.buffer = vdb.gpuBuffer,
			.byteOffset = 0u,
			.data = srat::slice<u8 const>(vdb.blob.data(), vdb.blob.size()),
		});
		// createNanoGrid computed per-grid stats during conversion; the
		// root's maximum is the majorant input for delta/ratio tracking
		auto const * nanoGrid = (
			reinterpret_cast<nanovdb::FloatGrid const *>(vdb.blob.data())
		);
		vdb.densityMax = nanoGrid->tree().root().maximum();
		// must mirror VdbHandle in shaders/util-vdb.glsl (scalar layout)
		struct GpuVdbHandle {
			u64 vdbGrid;
			f32 densityMax;
			f32 pad0;
		};
		GpuVdbHandle const handle = {
			.vdbGrid = vkof::buffer_virtual_address(vdb.gpuBuffer),
			.densityMax = vdb.densityMax,
			.pad0 = 0.0f,
		};
		vdb.handleBuffer = vkof::buffer_create({
			.byteCount = sizeof(GpuVdbHandle),
			.memory = vkof::BufferMemory::DeviceOnly,
		});
		vkof::buffer_upload({
			.buffer = vdb.handleBuffer,
			.byteOffset = 0u,
			.data = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(&handle), sizeof(handle)
			),
		});
		break;
	}
	file.close();
	return vdb;
}
