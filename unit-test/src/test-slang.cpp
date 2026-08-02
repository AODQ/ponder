#include "util-slang.hpp"

#include <mor/mor-shared.h>

TEST_SUITE("[headless]") {

TEST_CASE("slang: harness self-test") {
	test::SlangTestResult const result = test::slang_test_run({
		.file = "harness.slang",
		.threadCount = 100u,
	});
	CHECK(result.checkCount == 100u * 5u);
	CHECK(result.failCount == 0u);
}

TEST_CASE("slang: harness reports failures") {
	char const * const defines[] = { "HARNESS_EXPECT_FAIL" };
	test::SlangTestResult const result = test::slang_test_run({
		.file = "harness.slang",
		.threadCount = 3u,
		.defines = srat::slice<char const * const>(defines, 1u),
		.reportFailures = false,
	});
	REQUIRE(result.compiled);
	CHECK(result.checkCount == 3u * 6u);
	REQUIRE(result.failCount == 3u);
	REQUIRE(result.failures.size() == 3u);
	test::SlangTestFailure const & f = result.failures[0];
	CHECK(f.componentCount == 3u);
	CHECK(f.actual[1] == 2.0f);
	CHECK(f.expected[1] == 2.5f);
	CHECK(f.line > 0u);
}

TEST_CASE("slang: manifold constraint and solver") {
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	test::slang_test_run({
		.file = "manifold.slang",
		.threadCount = 256u,
		.includePaths = srat::slice<char const * const>(includePaths, 1u),
	});
}


namespace
{

// mirrors manifold.MeshData in util-manifold-mesh.slang
struct SlangMeshData {
	u64 positionVa;
	u64 attributeVa;
	u64 indexVa;
	f32m44 modelMatrix;
	u32 triangleCount;
	u32 pad;
};
static_assert(sizeof(SlangMeshData) == 96u);

f32m44 const skIdentity = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
};

}

// ---------------------------------------------------------------------------
// the mesh surface walks real geometry: a step can leave the current
// triangle, so advance() reprojects with a ray cast and re-fetches whatever
// triangle it lands on. the sphere mesh gives it curvature to walk over and
// a known analytic surface to check against
// ---------------------------------------------------------------------------

TEST_CASE("slang: manifold mesh surface") {
	f32 const radius = 1.0f;
	test::Mesh const mesh = test::make_sphere_mesh(radius, 32u, 64u);

	// analytic sphere normals in mor's interleaved attribute layout; the walk
	// needs a smooth field, and a faceted one would leave the solver a
	// singular jacobian
	std::vector<GpuMorVertexAttribute> attributes(mesh.vertices.size());
	for (usize i = 0u; i < mesh.vertices.size(); ++i) {
		test::Vertex const & v = mesh.vertices[i];
		f32 const len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
		attributes[i] = GpuMorVertexAttribute {
			.normal = f32v3 { v.x / len, v.y / len, v.z / len },
			.uv = f32v2 { 0.0f, 0.0f },
			.tangent = f32v4 { 1.0f, 0.0f, 0.0f, 1.0f },
		};
	}

	std::vector<test::Mesh> const meshes { mesh };
	std::vector<test::GeometryInstance> instances;
	instances.emplace_back(test::GeometryInstance {
		.meshIndex = 0u,
		.transform = skIdentity,
		.customIndex = 0u,
	});
	test::GeometryScene const scene = test::geometry_scene_create(
		meshes, instances
	);
	vkof::acceleration_structure_set_tlas(scene.tlas);

	vkof::Buffer const attributeBuf = test::upload_bytes(
		attributes.data(), attributes.size() * sizeof(GpuMorVertexAttribute)
	);

	SlangMeshData const meshData {
		.positionVa = vkof::buffer_virtual_address(scene.vertexBuffers[0]),
		.attributeVa = vkof::buffer_virtual_address(attributeBuf),
		.indexVa = vkof::buffer_virtual_address(scene.indexBuffers[0]),
		.modelMatrix = skIdentity,
		.triangleCount = static_cast<u32>(mesh.indices.size() / 3u),
		.pad = 0u,
	};
	vkof::Buffer const meshDataBuf = test::upload_bytes(
		&meshData, sizeof(meshData)
	);

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	test::slang_test_run({
		.file = "surface-mesh.slang",
		.threadCount = 256u,
		.includePaths = srat::slice<char const * const>(includePaths, 1u),
		.userVa = vkof::buffer_virtual_address(meshDataBuf),
	});

	vkof::buffer_destroy(meshDataBuf);
	vkof::buffer_destroy(attributeBuf);
	test::geometry_scene_destroy(scene);
}

}
