#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

TEST_SUITE("[headless]") {

TEST_CASE("pipeline: compute create valid shader") {
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fill_u32.comp",
	});
	CHECK(pl.id != 0);
	vkof::pipeline_destroy(pl);
}

TEST_CASE("pipeline: compute create passthrough shader") {
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "passthrough.comp",
	});
	CHECK(pl.id != 0);
	vkof::pipeline_destroy(pl);
}

TEST_CASE("pipeline: destroy does not crash") {
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fill_u32.comp",
	});
	REQUIRE(pl.id != 0);
	vkof::pipeline_destroy(pl);
	// double-destroy: handle is gone from pool, should be silently ignored
	vkof::pipeline_destroy(pl);
}

TEST_CASE("pipeline: two compute pipelines have distinct ids") {
	auto p0 = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fill_u32.comp",
	});
	auto p1 = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "passthrough.comp",
	});
	REQUIRE(p0.id != 0);
	REQUIRE(p1.id != 0);
	CHECK(p0.id != p1.id);
	vkof::pipeline_destroy(p0);
	vkof::pipeline_destroy(p1);
}

TEST_CASE("pipeline: re-create after destroy") {
	auto p0 = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fill_u32.comp",
	});
	REQUIRE(p0.id != 0);
	vkof::pipeline_destroy(p0);

	auto p1 = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "fill_u32.comp",
	});
	CHECK(p1.id != 0);
	vkof::pipeline_destroy(p1);
}

TEST_CASE("pipeline: graphics create") {
	static constexpr vkof::ImageFormat kColorFmts[] = {
		vkof::ImageFormat::r8g8b8a8_unorm,
	};
	auto pl = vkof::pipeline_graphics_create({
		.pathMesh = TEST_SHADER_DIR "test_minimal.mesh",
		.pathFragment = TEST_SHADER_DIR "test_minimal.frag",
		.attachmentColorFormats = srat::slice<vkof::ImageFormat const>(kColorFmts, 1u),
		.attachmentDepthStencilFormat = vkof::ImageFormat::none,
		.depthTest = vkof::DepthTest::write_off_test_off,
		.cullMode = vkof::CullMode::none,
		.blendMode = vkof::BlendMode::none,
	});
	CHECK(pl.id != 0);
	vkof::pipeline_destroy(pl);
}

TEST_CASE("pipeline: rapid compute create/destroy cycle") {
	for (u32 i = 0u; i < 32u; i++) {
		vkof::Pipeline const pl = vkof::pipeline_compute_create({
			.pathCompute = TEST_SHADER_DIR "passthrough.comp",
		});
		CHECK(pl.id != 0u);
		vkof::pipeline_destroy(pl);
	}
}

} // TEST_SUITE("[headless]")
