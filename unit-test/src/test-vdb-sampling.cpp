#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <ponder/vdb.hpp>
#include "util.hpp"

#include <nanovdb/NanoVDB.h>

#include <filesystem>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// vdb density crosscheck: loads a real .vdb through the production loader
// (ponder::vdb_load -- openvdb file -> createNanoGrid -> gpu upload), then
// samples utilVdbDensity on the gpu at a few thousand world positions and
// compares against cpu-side nanovdb accessor reads of the byte-identical
// blob. both sides read the same serialized bytes, so agreement is expected
// bit-exact -- this pins down the locally patched PNanoVDB.h (the glslang
// constructor-table rewrite touched the offset tables all traversal depends
// on; a transposed constant would corrupt every lookup) and the
// world-to-index transform path in util-vdb.glsl.
//
// sample points sit at voxel centers: the gpu re-derives the voxel from the
// world position via world_to_index + floor, and centers are maximally far
// from the floor() boundaries, so cpu/gpu fma rounding differences in the
// affine map can't land the two sides in different voxels. points are drawn
// from the index bbox plus a margin so the background (zero, outside active
// topology) path is exercised alongside real density reads.
// ---------------------------------------------------------------------------

namespace {

// deterministic lcg so the sampled voxel set is stable run to run
u32 lcg_next(u32 & state) {
	state = state * 1664525u + 1013904223u;
	return state;
}

} // namespace

TEST_SUITE("[headless]") {

TEST_CASE("vdb density: gpu sampling matches cpu nanovdb accessor") {
	std::string const vdbPath = (
		std::string(REPO_DIR) + "/assets/vdb/bunny_cloud.vdb"
	);
	REQUIRE(std::filesystem::exists(vdbPath));

	ponder::vdb_initialize();
	ponder::Vdb const vdb = ponder::vdb_load(vdbPath.c_str());
	REQUIRE(vdb.gpuBuffer.id != 0u);
	REQUIRE(!vdb.blob.empty());

	auto const * grid = (
		reinterpret_cast<nanovdb::FloatGrid const *>(vdb.blob.data())
	);
	REQUIRE(grid->isValid());
	REQUIRE(grid->gridType() == nanovdb::GridType::Float);
	REQUIRE(grid->activeVoxelCount() > 0u);
	// the loader reads this from the grid stats for the tracking majorant;
	// it must bound every density the sweep below observes
	REQUIRE(vdb.densityMax > 0.0f);
	REQUIRE(vdb.handleBuffer.id != 0u);
	MESSAGE(
		"bunny_cloud: ", vdb.blob.size(), " bytes, ",
		grid->activeVoxelCount(), " active voxels"
	);

	nanovdb::CoordBBox const bbox = grid->indexBBox();
	nanovdb::Coord const dim = bbox.dim();
	// margin of voxels beyond the index bbox so a slice of the samples reads
	// the background-value path, not just active topology
	i32 const margin = 8;

	u32 const sampleCount = 4096u;
	std::vector<f32> points(3u * sampleCount);
	std::vector<f32> expected(sampleCount);
	u32 nonzeroExpected = 0u;

	u32 rng = 0x12345678u;
	auto acc = grid->getAccessor();
	for (u32 i = 0u; i < sampleCount; ++i) {
		i32 const x = (
			bbox.min()[0] - margin
			+ (i32)(lcg_next(rng) % (u32)(dim[0] + 2 * margin))
		);
		i32 const y = (
			bbox.min()[1] - margin
			+ (i32)(lcg_next(rng) % (u32)(dim[1] + 2 * margin))
		);
		i32 const z = (
			bbox.min()[2] - margin
			+ (i32)(lcg_next(rng) % (u32)(dim[2] + 2 * margin))
		);
		nanovdb::Coord const ijk(x, y, z);
		expected[i] = acc.getValue(ijk);
		if (expected[i] != 0.0f) { nonzeroExpected++; }
		nanovdb::Vec3f const world = grid->indexToWorldF(nanovdb::Vec3f(
			(f32)x + 0.5f, (f32)y + 0.5f, (f32)z + 0.5f
		));
		points[3u * i + 0u] = world[0];
		points[3u * i + 1u] = world[1];
		points[3u * i + 2u] = world[2];
	}
	// canary against a trivially-green all-zero comparison: the sample set
	// must actually cover a meaningful amount of active topology
	REQUIRE(nonzeroExpected > 100u);

	auto const pointsBuf = test::upload_bytes(
		points.data(), points.size() * sizeof(f32)
	);
	auto const outBuf = vkof::buffer_create({
		.byteCount = (u64)sampleCount * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	// util-vdb.glsl lives in lib/ponder/shaders; the app dir stays on the
	// path for shared/global_pc.h
	char const * const includePaths[] = { APP_SHADER_DIR, PONDER_SHADER_DIR };
	auto const pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "vdb_sampling.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 2u),
	});
	REQUIRE(pl.id != 0u);

	struct VdbSamplingPush {
		u64 vdbVa;
		u64 pointsVa;
		u64 outVa;
		u32 count;
	};
	VdbSamplingPush const push {
		.vdbVa = vkof::buffer_virtual_address(vdb.gpuBuffer),
		.pointsVa = vkof::buffer_virtual_address(pointsBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.count = sampleCount,
	};
	test::dispatch(pl, push, (sampleCount + 63u) / 64u);
	test::gpu_wait();

	std::vector<f32> const gpuValues = (
		test::readback<f32>(outBuf, 0u, sampleCount)
	);

	// both sides read the same serialized float, so equality is bit-exact;
	// any mismatch means traversal/offset breakage, not float noise
	u32 mismatchCount = 0u;
	for (u32 i = 0u; i < sampleCount; ++i) {
		if (gpuValues[i] != expected[i]) {
			if (mismatchCount < 8u) {
				MESSAGE(
					"mismatch at sample ", i, ": gpu=", gpuValues[i],
					" cpu=", expected[i],
					" world=(", points[3u * i + 0u], ", ",
					points[3u * i + 1u], ", ", points[3u * i + 2u], ")"
				);
			}
			mismatchCount++;
		}
	}
	CHECK(mismatchCount == 0u);
	MESSAGE(
		"vdb crosscheck: ", sampleCount, " samples, ",
		nonzeroExpected, " nonzero, ", mismatchCount, " mismatches"
	);

	vkof::pipeline_destroy(pl);
	vkof::buffer_destroy(outBuf);
	vkof::buffer_destroy(pointsBuf);
	vkof::buffer_destroy(vdb.handleBuffer);
	vkof::buffer_destroy(vdb.gpuBuffer);
}

} // TEST_SUITE("[headless]")
