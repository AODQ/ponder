#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"
#include <vector>

// A simple triangle at z=1, centered on the XY origin.
// Positions are tightly-packed f32v3 as required by blas_create.
struct Vertex { f32 x, y, z; };

static Vertex const skTriVerts[] = {
	{ -0.5f, -0.5f, 1.0f },
	{  0.5f, -0.5f, 1.0f },
	{  0.0f,  0.5f, 1.0f },
};
static u32 const skTriIndices[] = { 0u, 1u, 2u };

static f32m44 const skIdentity = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
};

static vkof::Buffer upload_vertices() {
	auto buf = vkof::buffer_create({
		.byteCount = sizeof(skTriVerts),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	vkof::buffer_upload({
		.buffer = buf,
		.byteOffset = 0u,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(skTriVerts),
			sizeof(skTriVerts)
		),
	});
	return buf;
}

static vkof::Buffer upload_indices() {
	auto buf = vkof::buffer_create({
		.byteCount = sizeof(skTriIndices),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	vkof::buffer_upload({
		.buffer = buf,
		.byteOffset = 0u,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(skTriIndices),
			sizeof(skTriIndices)
		),
	});
	return buf;
}

// Build a TLAS containing one instance of the given BLAS.
static void build_tlas(
	vkof::AccelerationStructureTlas tlas,
	vkof::AccelerationStructureBlas blas
) {
	vkof::TlasInstance const inst {
		.blas = blas,
		.transform = skIdentity,
		.instanceCustomIndex = 0u,
		.rayMask = 0xFFu,
	};
	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::compute }
	);
	vkof::render_node_callback({
		.node = node,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			vkof::tlas_build(
				cmd, tlas, srat::slice<vkof::TlasInstance const>(&inst, 1u)
			);
		},
	});
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(&node, 1u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(node);
	test::gpu_wait();
}

TEST_SUITE("[headless]") {

TEST_CASE("accel: blas_create single triangle") {
	auto vb = upload_vertices();
	auto ib = upload_indices();

	auto blas = vkof::blas_create({
		.positionVa = vkof::buffer_virtual_address(vb),
		.vertexCount = 3u,
		.indexVa = vkof::buffer_virtual_address(ib),
		.triangleCount = 1u,
	});
	CHECK(blas.id != 0);

	vkof::blas_destroy(blas);
	vkof::buffer_destroy(ib);
	vkof::buffer_destroy(vb);
}

TEST_CASE("accel: tlas_create") {
	auto tlas = vkof::tlas_create({ .maxInstances = 1u });
	CHECK(tlas.id != 0);
	vkof::tlas_destroy(tlas);
}

TEST_CASE("accel: tlas_build one instance") {
	auto vb = upload_vertices();
	auto ib = upload_indices();

	auto blas = vkof::blas_create({
		.positionVa = vkof::buffer_virtual_address(vb),
		.vertexCount = 3u,
		.indexVa = vkof::buffer_virtual_address(ib),
		.triangleCount = 1u,
	});
	REQUIRE(blas.id != 0);

	auto tlas = vkof::tlas_create({ .maxInstances = 1u });
	REQUIRE(tlas.id != 0);

	build_tlas(tlas, blas);

	CHECK(vkof::acceleration_structure_device_address(tlas) != 0u);

	vkof::tlas_destroy(tlas);
	vkof::blas_destroy(blas);
	vkof::buffer_destroy(ib);
	vkof::buffer_destroy(vb);
}

TEST_CASE("accel: ray query hit and miss") {
	auto vb = upload_vertices();
	auto ib = upload_indices();

	auto blas = vkof::blas_create({
		.positionVa = vkof::buffer_virtual_address(vb),
		.vertexCount = 3u,
		.indexVa = vkof::buffer_virtual_address(ib),
		.triangleCount = 1u,
	});
	REQUIRE(blas.id != 0);

	auto tlas = vkof::tlas_create({ .maxInstances = 1u });
	REQUIRE(tlas.id != 0);

	build_tlas(tlas, blas);
	vkof::acceleration_structure_set_tlas(tlas);

	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "ray_query.comp",
	});
	REQUIRE(pl.id != 0);

	auto outBuf = vkof::buffer_create({
		.byteCount = 2u * sizeof(u32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push {
		u64 outVa;
	};
	Push const push { .outVa = vkof::buffer_virtual_address(outBuf) };
	test::dispatch(pl, push, /*groupCountX=*/1u);
	test::gpu_wait();

	auto const result = test::readback<u32>(outBuf, 0u, 2u);
	CHECK(result[0] == 1u); // invocation 0: ray toward triangle → hit
	CHECK(result[1] == 0u); // invocation 1: ray away from triangle → miss

	vkof::pipeline_destroy(pl);
	vkof::buffer_destroy(outBuf);
	vkof::tlas_destroy(tlas);
	vkof::blas_destroy(blas);
	vkof::buffer_destroy(ib);
	vkof::buffer_destroy(vb);
}

} // TEST_SUITE("[headless]")
