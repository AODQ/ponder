// unit-test/test_rendernode.cpp

#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

struct FillPush {
	u64 dstVa;
	u32 value;
	u32 count;
};

static constexpr u32 kLocalSize = 64;
static u32 groups_for(u32 count) {
	return (count + kLocalSize - 1) / kLocalSize;
}

TEST_SUITE("[headless]") {

TEST_CASE("rendernode: create/destroy") {
	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::compute }
	);
	CHECK(node.id != 0);
	vkof::render_node_destroy(node);
}

TEST_CASE("rendernode: graphics queue create/destroy") {
	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::graphics }
	);
	CHECK(node.id != 0);
	vkof::render_node_destroy(node);
}

TEST_CASE("rendernode: execute empty node list") {
	vkof::render_graph_execute({
		.nodes			= srat::slice<vkof::RenderNode const>(nullptr, 0),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0),
	});
	test::gpu_wait();
}

TEST_CASE("rendernode: compute callback executes on GPU") {
	constexpr u32 kCount = 256;
	constexpr u32 kValue = 0xABCD1234u;

	auto pl  = vkof::pipeline_compute_create({ .pathCompute = TEST_SHADER_DIR "fill_u32.comp" });
	auto buf = test::make_buffer_u32(kCount);
	REQUIRE(pl.id != 0);

	FillPush const push {
		.dstVa = vkof::buffer_virtual_address(buf),
		.value  = kValue,
		.count  = kCount,
	};
	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::compute }
	);
	vkof::render_node_callback({
		.node = node,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			vkof::cmd_dispatch({
				.cmd = cmd,
				.pipeline = pl,
				.pushconstant = srat::slice_as_bytes(push),
				.threadgroupSize = u32v3{ kLocalSize, 1u, 1u },
				.invocationCount = u32v3{ kCount, 1u, 1u },
			});
		},
	});

	vkof::render_graph_execute({
		.nodes			= srat::slice<vkof::RenderNode const>(&node, 1),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0),
	});
	vkof::render_node_destroy(node);

	auto result = test::readback<u32>(buf, 0, kCount);
	for (u32 i = 0; i < kCount; ++i) {
		CHECK(result[i] == kValue);
	}

	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
}

TEST_CASE("rendernode: add_image barrier") {
	auto ti = vkof::transient_image_create({
		.format		   = vkof::ImageFormat::r8g8b8a8_unorm,
		.scaleWidth	   = 0.5f,
		.scaleHeight	  = 0.5f,
		.mipLevels		= 1,
		.isDoubleBuffered = false,
	});

	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::compute }
	);
	vkof::render_node_add_image({
		.node   = node,
		.image  = ti,
		.access = vkof::RenderNodeAccess::write,
	});
	vkof::render_graph_execute({
		.nodes			= srat::slice<vkof::RenderNode const>(&node, 1),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0),
	});
	vkof::render_node_destroy(node);
	test::gpu_wait();
}

TEST_CASE("rendernode: attachment_color clear") {
	auto ti = vkof::transient_image_create({
		.format = vkof::ImageFormat::r8g8b8a8_unorm,
		.scaleWidth = 0.25f,
		.scaleHeight = 0.25f,
		.mipLevels = 1,
		.isDoubleBuffered = false,
	});
	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::graphics }
	);
	f32 const clearColor[] = { 1.0f, 0.0f, 0.0f, 1.0f };
	vkof::render_node_attachment_color({
		.node = node,
		.image = ti,
		.loadOp = vkof::RenderNodeLoadOp::clear,
		.mipLevel = 0u,
		.colorIndex = 0u,
		.clearColor = srat::slice<f32 const>(clearColor, 4u),
	});
	vkof::render_node_callback({
		.node = node,
		.callback = [](vkof::CommandBuffer const &) {},
	});
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(&node, 1u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(node);
	test::gpu_wait();
}

TEST_CASE("rendernode: attachment_depth clear") {
	auto depth = vkof::transient_image_create({
		.format = vkof::ImageFormat::d24_unorm_s8_uint,
		.scaleWidth = 0.25f,
		.scaleHeight = 0.25f,
		.mipLevels = 1,
		.isDoubleBuffered = false,
	});
	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::graphics }
	);
	f32 const clearDepth[] = { 1.0f };
	vkof::render_node_attachment_depth({
		.node = node,
		.image = depth,
		.loadOp = vkof::RenderNodeLoadOp::clear,
		.mipLevel = 0u,
		.clearDepth = srat::slice<f32 const>(clearDepth, 1u),
	});
	vkof::render_node_callback({
		.node = node,
		.callback = [](vkof::CommandBuffer const &) {},
	});
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(&node, 1u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(node);
	test::gpu_wait();
}

TEST_CASE("rendernode: persistent image declared") {
	auto img = vkof::image_create({
		.width = 64u,
		.height = 64u,
		.format = vkof::ImageFormat::r32_float,
		.mipLevels = 1u,
		.optInitialData = srat::slice<u8 const>(nullptr, 0u),
	});
	REQUIRE(img.id != 0);

	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::compute }
	);
	vkof::render_node_add_persistent_image({
		.node = node,
		.image = img,
		.access = vkof::RenderNodeAccess::write,
	});
	vkof::render_node_callback({
		.node = node,
		.callback = [](vkof::CommandBuffer const &) {},
	});
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(&node, 1u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(node);
	test::gpu_wait();
	vkof::image_destroy(img);
}

TEST_CASE("rendernode: two sequential executes") {
	constexpr u32 kCount = 64;

	auto pl  = vkof::pipeline_compute_create({ .pathCompute = TEST_SHADER_DIR "fill_u32.comp" });
	auto buf = test::make_buffer_u32(kCount);
	REQUIRE(pl.id != 0);

	for (u32 pass = 0; pass < 2; ++pass) {
		FillPush const push {
			.dstVa = vkof::buffer_virtual_address(buf),
			.value  = pass + 100u,
			.count  = kCount,
		};
		test::dispatch(pl, push, groups_for(kCount));
	}

	auto result = test::readback<u32>(buf, 0, kCount);
	for (u32 i = 0; i < kCount; ++i) {
		CHECK(result[i] == 101u); // last dispatch wins
	}

	vkof::buffer_destroy(buf);
	vkof::pipeline_destroy(pl);
}

} // TEST_SUITE("[headless]")
