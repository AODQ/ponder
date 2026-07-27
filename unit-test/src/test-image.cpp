#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

TEST_SUITE("[headless]") {

TEST_CASE("image: create/destroy r8g8b8a8_unorm") {
	auto img = vkof::image_create({
		.width = 64,
		.height= 64,
		.format= vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(img.id != 0);
	vkof::image_destroy(img);
}

TEST_CASE("image: width/height accessors") {
	auto img = vkof::image_create({
		.width = 128,
		.height = 256,
		.format = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(vkof::image_width(img) == 128u);
	CHECK(vkof::image_height(img) == 256u);
	vkof::image_destroy(img);
}

TEST_CASE("image: r16g16b16a16_sfloat format") {
	auto img = vkof::image_create({
		.width = 64,
		.height = 64,
		.format = vkof::ImageFormat::r16g16b16a16_sfloat,
		.mipLevels = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(img.id != 0);
	vkof::image_destroy(img);
}

TEST_CASE("image: r32_float format") {
	auto img = vkof::image_create({
		.width = 64,
		.height = 64,
		.format = vkof::ImageFormat::r32_float,
		.mipLevels = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(img.id != 0);
	vkof::image_destroy(img);
}

TEST_CASE("image: d24_unorm_s8_uint format") {
	auto img = vkof::image_create({
		.width = 64,
		.height = 64,
		.format = vkof::ImageFormat::d24_unorm_s8_uint,
		.mipLevels = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(img.id != 0);
	vkof::image_destroy(img);
}

TEST_CASE("image: mipLevels=4") {
	auto img = vkof::image_create({
		.width = 256,
		.height = 256,
		.format = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels = 4,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(img.id != 0);
	vkof::image_destroy(img);
}

TEST_CASE("image: image_sampler_handle non-zero") {
	auto img = vkof::image_create({
		.width = 64,
		.height = 64,
		.format = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	auto smp = vkof::sampler_create({
		.magFilter= vkof::SamplerFilter::nearest,
		.minFilter= vkof::SamplerFilter::nearest,
		.addressModeU = vkof::SamplerAddressMode::repeat,
		.addressModeV = vkof::SamplerAddressMode::repeat,
		.addressModeW = vkof::SamplerAddressMode::repeat,
	});
	CHECK(vkof::image_sampler_handle({ .image = img, .sampler = smp }) != 0u);
	vkof::sampler_destroy(smp);
	vkof::image_destroy(img);
}

TEST_CASE("image: image_storage_handle non-zero") {
	auto img = vkof::image_create({
		.width = 64,
		.height = 64,
		.format = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(
		vkof::image_storage_handle({ .image = img, .mipLevel = 0 }) != 0u
	);
	vkof::image_destroy(img);
}

TEST_CASE("image: storage handle slot is recycled after destroy") {
	auto const makeImageAndGetSlot = []() -> u32 {
		auto const img = vkof::image_create({
			.width = 64,
			.height = 64,
			.format = vkof::ImageFormat::r8g8b8a8_unorm,
			.mipLevels = 1,
			.optInitialData = srat::slice<u8 const>(nullptr, 0),
		});
		u32 const slot = (
			vkof::image_storage_handle({ .image = img, .mipLevel = 0 })
		);
		vkof::image_destroy(img);
		return slot;
	};

	u32 const firstSlot = makeImageAndGetSlot();
	u32 const secondSlot = makeImageAndGetSlot();
	CHECK(firstSlot == secondSlot);
}

TEST_CASE("image: 1x1 image") {
	auto img = vkof::image_create({
		.width = 1,
		.height = 1,
		.format = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(img.id != 0);
	CHECK(vkof::image_width(img) == 1u);
	CHECK(vkof::image_height(img) == 1u);
	vkof::image_destroy(img);
}

TEST_CASE("image: 1024x1024 image") {
	auto img = vkof::image_create({
		.width = 1024,
		.height = 1024,
		.format = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels = 1,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	CHECK(img.id != 0);
	vkof::image_destroy(img);
}

TEST_CASE("image: generate_mipmaps no crash") {
	auto img = vkof::image_create({
		.width = 256,
		.height = 256,
		.format = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels = 4,
		.optInitialData = srat::slice<u8 const>(nullptr, 0),
	});
	REQUIRE(img.id != 0);
	vkof::image_generate_mipmaps(img);
	test::gpu_wait();
	vkof::image_destroy(img);
}

TEST_CASE("image: initial data round-trip") {
	constexpr u32 kW = 4u;
	constexpr u32 kH = 4u;
	constexpr u32 kCount = kW * kH;

	std::vector<f32> pixels(kCount);
	for (u32 i = 0u; i < kCount; ++i) {
		pixels[i] = (f32)i / (f32)kCount;
	}

	auto img = vkof::image_create({
		.width = kW,
		.height = kH,
		.format = vkof::ImageFormat::r32_float,
		.mipLevels = 1,
		.optInitialData = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(pixels.data()),
			kCount * sizeof(f32)
		),
	});
	REQUIRE(img.id != 0);

	u32 const imgHandle = (
		vkof::image_storage_handle({ .image = img, .mipLevel = 0u })
	);
	REQUIRE(imgHandle != 0u);

	auto outBuf = vkof::buffer_create({
		.byteCount = kCount * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "image_read_r32.comp",
	});
	REQUIRE(pl.id != 0);

	struct Push {
		u64 dstVa;
		u32 imgHandle;
		u32 width;
		u32 height;
	};
	Push const push {
		.dstVa = vkof::buffer_virtual_address(outBuf),
		.imgHandle = imgHandle,
		.width = kW,
		.height = kH,
	};

	vkof::RenderNode const node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::compute }
	);
	vkof::render_node_add_persistent_image({
		.node = node,
		.image = img,
		.access = vkof::RenderNodeAccess::read,
	});
	vkof::render_node_callback({
		.node = node,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			vkof::cmd_dispatch({
				.cmd = cmd,
				.pipeline = pl,
				.pushconstant = srat::slice_as_bytes(push),
				.threadgroupSize = u32v3{ 64u, 1u, 1u },
				.invocationCount = u32v3{ kCount, 1u, 1u },
			});
		},
	});
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(&node, 1u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(node);

	auto const result = test::readback<f32>(outBuf, 0u, kCount);
	for (u32 i = 0u; i < kCount; ++i) {
		CAPTURE(i);
		CHECK(result[i] == doctest::Approx(pixels[i]).epsilon(0.001f));
	}

	vkof::pipeline_destroy(pl);
	vkof::buffer_destroy(outBuf);
	vkof::image_destroy(img);
}

TEST_CASE("image: rapid create/destroy cycle") {
	for (u32 i = 0u; i < 128u; i++) {
		vkof::Image const img = vkof::image_create({
			.width = 4u,
			.height = 4u,
			.format = vkof::ImageFormat::r8g8b8a8_unorm,
			.mipLevels = 1u,
			.optInitialData = {},
		});
		CHECK(img.id != 0u);
		vkof::image_destroy(img);
	}
}

TEST_CASE("image3d: create/destroy r16g16b16a16_sfloat") {
	vkof::Image const img = vkof::image_create({
		.width = 32u,
		.height = 32u,
		.depth = 8u,
		.format = vkof::ImageFormat::r16g16b16a16_sfloat,
		.mipLevels = 1u,
		.optInitialData = {},
	});
	CHECK(img.id != 0u);
	vkof::image_destroy(img);
}

TEST_CASE("image3d: depth accessor") {
	vkof::Image const img = vkof::image_create({
		.width = 16u,
		.height = 8u,
		.depth = 4u,
		.format = vkof::ImageFormat::r32_float,
		.mipLevels = 1u,
		.optInitialData = {},
	});
	REQUIRE(img.id != 0u);
	CHECK(vkof::image_width(img) == 16u);
	CHECK(vkof::image_height(img) == 8u);
	CHECK(vkof::image_depth(img) == 4u);
	vkof::image_destroy(img);
}

TEST_CASE("image3d: image_sampler3d_handle non-zero") {
	vkof::Image const img = vkof::image_create({
		.width = 8u,
		.height = 8u,
		.depth = 4u,
		.format = vkof::ImageFormat::r16g16b16a16_sfloat,
		.mipLevels = 1u,
		.optInitialData = {},
	});
	vkof::Sampler const smp = vkof::sampler_create({
		.magFilter = vkof::SamplerFilter::linear,
		.minFilter = vkof::SamplerFilter::linear,
		.addressModeU = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
	});
	CHECK(vkof::image_sampler3d_handle({ .image = img, .sampler = smp }) != 0u);
	vkof::sampler_destroy(smp);
	vkof::image_destroy(img);
}

TEST_CASE("image3d: image_storage3d_handle non-zero") {
	vkof::Image const img = vkof::image_create({
		.width = 8u,
		.height = 8u,
		.depth = 4u,
		.format = vkof::ImageFormat::r16g16b16a16_sfloat,
		.mipLevels = 1u,
		.optInitialData = {},
	});
	CHECK(vkof::image_storage3d_handle({ .image = img, .mipLevel = 0u }) != 0u);
	vkof::image_destroy(img);
}

TEST_CASE("image3d: imageStore/imageLoad round-trip") {
	constexpr u32 kW = 4u;
	constexpr u32 kH = 4u;
	constexpr u32 kD = 4u;
	constexpr u32 kTotal = kW * kH * kD;

	vkof::Image const img = vkof::image_create({
		.width = kW,
		.height = kH,
		.depth = kD,
		.format = vkof::ImageFormat::r32_float,
		.mipLevels = 1u,
		.optInitialData = {},
	});
	REQUIRE(img.id != 0u);

	u32 const storageHandle = (
		vkof::image_storage3d_handle({ .image = img, .mipLevel = 0u })
	);
	REQUIRE(storageHandle != 0u);

	vkof::Buffer const outBuf = vkof::buffer_create({
		.byteCount = kTotal * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	vkof::Pipeline const fillPl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "image3d_fill.comp",
	});
	vkof::Pipeline const readPl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "image3d_read.comp",
	});
	REQUIRE(fillPl.id != 0u);
	REQUIRE(readPl.id != 0u);

	struct FillPush { u32 imgHandle; u32 w; u32 h; u32 d; };
	struct ReadPush { u64 dstVa; u32 imgHandle; u32 w; u32 h; u32 d; };

	vkof::RenderNode const fillNode = (
		vkof::render_node_create({ .queue = vkof::CommandQueue::compute })
	);
	vkof::render_node_add_persistent_image({
		.node = fillNode,
		.image = img,
		.access = vkof::RenderNodeAccess::write,
	});
	vkof::render_node_callback({
		.node = fillNode,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			FillPush const push {
				.imgHandle = storageHandle,
				.w = kW, .h = kH, .d = kD,
			};
			vkof::cmd_dispatch({
				.cmd = cmd,
				.pipeline = fillPl,
				.pushconstant = srat::slice_as_bytes(push),
				.threadgroupSize = u32v3{ 64u, 1u, 1u },
				.invocationCount = u32v3{ kTotal, 1u, 1u },
			});
		},
	});

	vkof::RenderNode const readNode = (
		vkof::render_node_create({ .queue = vkof::CommandQueue::compute })
	);
	vkof::render_node_add_persistent_image({
		.node = readNode,
		.image = img,
		.access = vkof::RenderNodeAccess::read,
	});
	vkof::render_node_callback({
		.node = readNode,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			ReadPush const push {
				.dstVa = vkof::buffer_virtual_address(outBuf),
				.imgHandle = storageHandle,
				.w = kW, .h = kH, .d = kD,
			};
			vkof::cmd_dispatch({
				.cmd = cmd,
				.pipeline = readPl,
				.pushconstant = srat::slice_as_bytes(push),
				.threadgroupSize = u32v3{ 64u, 1u, 1u },
				.invocationCount = u32v3{ kTotal, 1u, 1u },
			});
		},
	});

	vkof::RenderNode const nodes[] = { fillNode, readNode };
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(nodes, 2u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(fillNode);
	vkof::render_node_destroy(readNode);

	auto const result = test::readback<f32>(outBuf, 0u, kTotal);
	for (u32 i = 0u; i < kTotal; ++i) {
		CAPTURE(i);
		f32 const expected = (f32)i / (f32)kTotal;
		CHECK(result[i] == doctest::Approx(expected).epsilon(0.001f));
	}

	vkof::pipeline_destroy(fillPl);
	vkof::pipeline_destroy(readPl);
	vkof::buffer_destroy(outBuf);
	vkof::image_destroy(img);
}

TEST_CASE("image3d: ddgi atlas probe-tile layout") {
	static constexpr u32 kCountX = 3u;
	static constexpr u32 kCountY = 2u;
	static constexpr u32 kCountZ = 4u;
	static constexpr u32 kTotalProbes = kCountX * kCountY * kCountZ;

	vkof::Image const img = vkof::image_create({
		.width = kCountX * 8u,
		.height = kCountY * 8u,
		.depth = kCountZ,
		.format = vkof::ImageFormat::r32_float,
		.mipLevels = 1u,
		.optInitialData = {},
	});
	REQUIRE(img.id != 0u);

	u32 const storageHandle = (
		vkof::image_storage3d_handle({ .image = img, .mipLevel = 0u })
	);
	REQUIRE(storageHandle != 0u);

	vkof::Buffer const outBuf = vkof::buffer_create({
		.byteCount = kTotalProbes * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	vkof::Pipeline const fillPl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "ddgi3d_probe_fill.comp",
	});
	vkof::Pipeline const readPl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "ddgi3d_probe_read.comp",
	});
	REQUIRE(fillPl.id != 0u);
	REQUIRE(readPl.id != 0u);

	struct FillPush { u32 imgHandle; u32 countX; u32 countY; u32 countZ; };
	struct ReadPush { u64 dstVa; u32 imgHandle; u32 countX; u32 countY; u32 countZ; };

	vkof::RenderNode const fillNode = (
		vkof::render_node_create({ .queue = vkof::CommandQueue::compute })
	);
	vkof::render_node_add_persistent_image({
		.node = fillNode,
		.image = img,
		.access = vkof::RenderNodeAccess::write,
	});
	vkof::render_node_callback({
		.node = fillNode,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			FillPush const push {
				.imgHandle = storageHandle,
				.countX = kCountX, .countY = kCountY, .countZ = kCountZ,
			};
			vkof::cmd_dispatch({
				.cmd = cmd,
				.pipeline = fillPl,
				.pushconstant = srat::slice_as_bytes(push),
				.threadgroupSize = u32v3{ 1u, 1u, 1u },
				.invocationCount = u32v3{ kCountX, kCountY, kCountZ },
			});
		},
	});

	vkof::RenderNode const readNode = (
		vkof::render_node_create({ .queue = vkof::CommandQueue::compute })
	);
	vkof::render_node_add_persistent_image({
		.node = readNode,
		.image = img,
		.access = vkof::RenderNodeAccess::read,
	});
	vkof::render_node_callback({
		.node = readNode,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			ReadPush const push {
				.dstVa = vkof::buffer_virtual_address(outBuf),
				.imgHandle = storageHandle,
				.countX = kCountX, .countY = kCountY, .countZ = kCountZ,
			};
			vkof::cmd_dispatch({
				.cmd = cmd,
				.pipeline = readPl,
				.pushconstant = srat::slice_as_bytes(push),
				.threadgroupSize = u32v3{ 64u, 1u, 1u },
				.invocationCount = u32v3{ kTotalProbes, 1u, 1u },
			});
		},
	});

	vkof::RenderNode const nodes[] = { fillNode, readNode };
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(nodes, 2u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(fillNode);
	vkof::render_node_destroy(readNode);

	auto const result = test::readback<f32>(outBuf, 0u, kTotalProbes);
	for (u32 pz = 0u; pz < kCountZ; ++pz) {
		for (u32 py = 0u; py < kCountY; ++py) {
			for (u32 px = 0u; px < kCountX; ++px) {
				u32 const idx = px + py * kCountX + pz * kCountX * kCountY;
				CAPTURE(px); CAPTURE(py); CAPTURE(pz);
				f32 const expected = (
					f32(pz) * 1000.0f + f32(px) * 10.0f + f32(py)
				);
				CHECK(result[idx] == doctest::Approx(expected).epsilon(0.01f));
			}
		}
	}

	vkof::pipeline_destroy(fillPl);
	vkof::pipeline_destroy(readPl);
	vkof::buffer_destroy(outBuf);
	vkof::image_destroy(img);
}

} // TEST_SUITE("[headless]")
