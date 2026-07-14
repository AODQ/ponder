// unit-test/test_transient.cpp

#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

TEST_SUITE("[headless]") {

TEST_CASE("transient: image create returns valid handle") {
	auto ti = vkof::transient_image_create({
		.format		   = vkof::ImageFormat::r8g8b8a8_unorm,
		.scaleWidth	   = 1.0f,
		.scaleHeight	  = 1.0f,
		.mipLevels		= 1,
		.isDoubleBuffered = false,
	});
	CHECK(ti.id != 0);
}

TEST_CASE("transient: get_image returns valid handle") {
	auto ti = vkof::transient_image_create({
		.format		   = vkof::ImageFormat::r8g8b8a8_unorm,
		.scaleWidth	   = 0.5f,
		.scaleHeight	  = 0.5f,
		.mipLevels		= 1,
		.isDoubleBuffered = false,
	});
	auto img = vkof::transient_image_get_image(ti);
	CHECK(img.id != 0);
}

TEST_CASE("transient: buffer create returns valid handle") {
	auto tb = vkof::transient_buffer_create({
		.byteCount		= 1024,
		.isDoubleBuffered = false,
	});
	CHECK(tb.id != 0);
}

TEST_CASE("transient: get_buffer returns valid handle") {
	auto tb = vkof::transient_buffer_create({
		.byteCount		= 1024,
		.isDoubleBuffered = false,
	});
	auto buf = vkof::transient_buffer_get_buffer(tb);
	CHECK(buf.id != 0);
}

TEST_CASE("transient: double-buffered image has valid handles") {
	auto ti = vkof::transient_image_create({
		.format		   = vkof::ImageFormat::r8g8b8a8_unorm,
		.scaleWidth	   = 1.0f,
		.scaleHeight	  = 1.0f,
		.mipLevels		= 1,
		.isDoubleBuffered = true,
	});
	// Get image on even frame (slot 0)
	auto img0 = vkof::transient_image_get_image(ti);
	CHECK(img0.id != 0);
	// Simulate a frame advance by calling render_graph_execute once (empty pass)
	{
		vkof::RenderNode emptyNode = vkof::render_node_create(
			{ .queue = vkof::CommandQueue::compute }
		);
		vkof::render_graph_execute({
			.nodes			= srat::slice<vkof::RenderNode const>(&emptyNode, 1),
			.rootPushconstant = srat::slice<u8 const>(nullptr, 0),
		});
		vkof::render_node_destroy(emptyNode);
		test::gpu_wait();
	}
	// Get image on odd frame (slot 1)
	auto img1 = vkof::transient_image_get_image(ti);
	CHECK(img1.id != 0);
}

TEST_CASE("transient: double-buffered storage handle differs per frame slot") {
	auto ti = vkof::transient_image_create({
		.format		   = vkof::ImageFormat::r8g8b8a8_unorm,
		.scaleWidth	   = 1.0f,
		.scaleHeight	  = 1.0f,
		.mipLevels		= 1,
		.isDoubleBuffered = true,
	});

	u32 const handleFrame0First = (
		vkof::transient_image_storage_handle({ .image = ti, .mipLevel = 0 })
	);
	u32 const handleFrame0Second = (
		vkof::transient_image_storage_handle({ .image = ti, .mipLevel = 0 })
	);
	// same physical image queried twice in the same frame -> same slot
	CHECK(handleFrame0First == handleFrame0Second);

	{
		vkof::RenderNode emptyNode = vkof::render_node_create(
			{ .queue = vkof::CommandQueue::compute }
		);
		vkof::render_graph_execute({
			.nodes			= srat::slice<vkof::RenderNode const>(&emptyNode, 1),
			.rootPushconstant = srat::slice<u8 const>(nullptr, 0),
		});
		vkof::render_node_destroy(emptyNode);
		test::gpu_wait();
	}

	u32 const handleFrame1 = (
		vkof::transient_image_storage_handle({ .image = ti, .mipLevel = 0 })
	);
	// other ping-pong image -> its own slot, not the frame-0 slot repointed
	CHECK(handleFrame1 != handleFrame0First);
}

} // TEST_SUITE("[headless]")
