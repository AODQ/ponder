#pragma once

// ---------------------------------------------------------------------------
// simple testing utilities that extend vkof
// ---------------------------------------------------------------------------

#include <vkof/vkof.hpp>
#include <srat/core-array.hpp>
#include <srat/core-types.hpp>
#include <vector>
#include <cstring>

#include <stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace test {

// dispatch a compute render node and return immediately
template <typename Push>
inline void dispatch(
	vkof::Pipeline pipeline,
	Push const & push,
	u32 groupX,
	u32 groupY = 1u,
	u32 groupZ = 1u,
	srat::slice<u8 const> rootPushconstant = srat::slice<u8 const>(nullptr, 0)
) {
	vkof::RenderNode node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::compute }
	);
	vkof::render_node_callback({
		.node = node,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			vkof::cmd_dispatch({
				.cmd = cmd,
				.pipeline = pipeline,
				.pushconstant = srat::slice_as_bytes(push),
				.threadgroupSize = u32v3{ 1u, 1u, 1u },
				.invocationCount = u32v3{ groupX, groupY, groupZ },
			});
		},
	});
	vkof::render_graph_execute({
		.nodes			= srat::slice<vkof::RenderNode const>(&node, 1),
		.rootPushconstant = rootPushconstant,
	});
	vkof::render_node_destroy(node);
}

// blocks until gpu has completed buffer download
template <typename T>
inline std::vector<T> readback(
	vkof::Buffer buf,
	u64  byteOffset,
	u32  count
) {
	std::vector<T> result(count);
	vkof::buffer_download({
		.buffer	 = buf,
		.byteOffset = byteOffset,
		.dst		= srat::slice<u8>(
			reinterpret_cast<u8 *>(result.data()),
			static_cast<u64>(count) * sizeof(T)
		),
	});
	return result;
}

// idle gpu, kind of hacky by forcing a buffer download
inline void gpu_wait() {
	auto scratch = vkof::buffer_create({
		.byteCount = 4,
		.memory	= vkof::BufferMemory::DeviceOnly,
	});
	u8 tmp[4] {};
	vkof::buffer_download({
		.buffer	 = scratch,
		.byteOffset = 0,
		.dst		= srat::slice<u8>(tmp, 4),
	});
	vkof::buffer_destroy(scratch);
}

// create a device-local buffer of `count` u32s
inline vkof::Buffer make_buffer_u32(u32 count) {
	return vkof::buffer_create({
		.byteCount = static_cast<u64>(count) * sizeof(u32),
		.memory	= vkof::BufferMemory::DeviceOnly,
	});
}

// writes an rgb png; non-finite values render as pure red, everything else
// clamped to [0,1] and scaled to a byte. returns false if the png write
// failed. asserts nothing itself -- caller decides how to treat nanCount.
inline bool write_heatmap_png(
	std::vector<f32> const & r,
	std::vector<f32> const & g,
	std::vector<f32> const & b,
	u32 width, u32 height, char const * path,
	u32 * outNanCount = nullptr
) {
	std::vector<u8> pixels(width * height * 3, 0);
	u32 nanCount = 0;
	for (u32 i = 0; i < width * height; ++i) {
		bool const finite = (
			std::isfinite(r[i]) && std::isfinite(g[i]) && std::isfinite(b[i])
		);
		if (!finite) {
			nanCount++;
			pixels[i*3+0] = 255; pixels[i*3+1] = 0; pixels[i*3+2] = 0;
		} else {
			pixels[i*3+0] = (u8)(std::clamp(r[i], 0.0f, 1.0f) * 255.0f);
			pixels[i*3+1] = (u8)(std::clamp(g[i], 0.0f, 1.0f) * 255.0f);
			pixels[i*3+2] = (u8)(std::clamp(b[i], 0.0f, 1.0f) * 255.0f);
		}
	}
	if (outNanCount) { *outNanCount = nanCount; }

	std::filesystem::path const p(path);
	std::filesystem::create_directories(p.parent_path());
	return stbi_write_png(
		path, (int)width, (int)height, 3, pixels.data(), (int)width * 3
	) != 0;
}

} // namespace test
