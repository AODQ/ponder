#pragma once

// ---------------------------------------------------------------------------
// simple testing utilities that extend vkof
// ---------------------------------------------------------------------------

#include <vkof/vkof.hpp>
#include <srat/core-array.hpp>
#include <srat/core-types.hpp>
#include <vector>
#include <cstring>

#include <doctest/doctest.h>
#include <stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace test {

// set by main.cpp when vkof-test is launched with --furnace; gates the
// opt-in full assets/Models furnace sweep in test-furnace-model-render.cpp
// (159 models is far too slow for the default suite run)
extern bool furnaceSweepEnabled;

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
	// probe_message is no longer auto-cleared by render_graph_execute (see
	// vkof.hpp), so an earlier, already-handled test case's leftover
	// message would otherwise leak into this dispatch's probe_message_count
	// check; reset here gives every dispatch() call the same clean-slate
	// window the old auto-clear implicitly provided
	vkof::probe_reset();
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

// uploads a device-local buffer initialized with data; shared by every test
// that builds its own blas/tlas geometry
inline vkof::Buffer upload_bytes(void const * data, u64 const byteCount) {
	auto buf = vkof::buffer_create({
		.byteCount = byteCount,
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	vkof::buffer_upload({
		.buffer = buf,
		.byteOffset = 0u,
		.data = srat::slice<u8 const>(
			reinterpret_cast<u8 const *>(data), byteCount
		),
	});
	return buf;
}

struct Vertex { f32 x, y, z; };

struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<u32> indices;
};

// uv sphere, `rings` latitude bands by `segments` longitude slices. radius
// is baked directly into the vertex positions rather than left to a tlas
// instance scale: geometric normals only transform correctly by the
// rotation part of an instance transform (mat3(objectToWorld) * geomN, see
// cornell_glass_render.comp), so any non-uniform or per-axis scale has to
// live in the mesh itself, and baking it in uniformly here keeps every
// instance of a sphere mesh translation/rotation-only
inline Mesh make_sphere_mesh(
	f32 const radius, u32 const rings = 12u, u32 const segments = 24u
) {
	Mesh mesh;
	f32 const pi = 3.14159265358979f;
	for (u32 ring = 0u; ring <= rings; ++ring) {
		f32 const theta = pi * static_cast<f32>(ring) / static_cast<f32>(rings);
		f32 const y = std::cos(theta);
		f32 const r = std::sin(theta);
		for (u32 seg = 0u; seg <= segments; ++seg) {
			f32 const phi = (
				2.0f * pi * static_cast<f32>(seg) / static_cast<f32>(segments)
			);
			mesh.vertices.push_back(Vertex {
				radius * r * std::cos(phi),
				radius * y,
				radius * r * std::sin(phi),
			});
		}
	}
	u32 const stride = segments + 1u;
	for (u32 ring = 0u; ring < rings; ++ring) {
		for (u32 seg = 0u; seg < segments; ++seg) {
			u32 const a = ring * stride + seg;
			u32 const b = a + stride;
			mesh.indices.push_back(a);
			mesh.indices.push_back(b);
			mesh.indices.push_back(a + 1u);
			mesh.indices.push_back(a + 1u);
			mesh.indices.push_back(b);
			mesh.indices.push_back(b + 1u);
		}
	}
	return mesh;
}

struct GeometryInstance {
	u32 meshIndex;
	f32m44 transform;
	u32 customIndex;
	u32 rayMask { 0xFFu };
};

struct GeometryScene {
	std::vector<vkof::Buffer> vertexBuffers;
	std::vector<vkof::Buffer> indexBuffers;
	std::vector<vkof::AccelerationStructureBlas> blases;
	vkof::AccelerationStructureTlas tlas;
};

// builds one blas per mesh and one tlas instance per GeometryInstance,
// shared by every test scene that instances a handful of mesh shapes many
// times over (checker floor tiles, grids of spheres, ...)
inline GeometryScene geometry_scene_create(
	std::vector<Mesh> const & meshes,
	std::vector<GeometryInstance> const & instances
) {
	GeometryScene scene {};
	scene.vertexBuffers.resize(meshes.size());
	scene.indexBuffers.resize(meshes.size());
	scene.blases.resize(meshes.size());

	for (u32 m = 0u; m < meshes.size(); ++m) {
		scene.vertexBuffers[m] = upload_bytes(
			meshes[m].vertices.data(),
			meshes[m].vertices.size() * sizeof(Vertex)
		);
		scene.indexBuffers[m] = upload_bytes(
			meshes[m].indices.data(),
			meshes[m].indices.size() * sizeof(u32)
		);
		scene.blases[m] = vkof::blas_create({
			.positionVa = vkof::buffer_virtual_address(scene.vertexBuffers[m]),
			.vertexCount = static_cast<u32>(meshes[m].vertices.size()),
			.indexVa = vkof::buffer_virtual_address(scene.indexBuffers[m]),
			.triangleCount = static_cast<u32>(meshes[m].indices.size() / 3u),
		});
		REQUIRE(scene.blases[m].id != 0);
	}

	scene.tlas = vkof::tlas_create({
		.maxInstances = static_cast<u32>(instances.size())
	});
	REQUIRE(scene.tlas.id != 0);

	std::vector<vkof::TlasInstance> tlasInstances(instances.size());
	for (u32 i = 0u; i < instances.size(); ++i) {
		tlasInstances[i] = vkof::TlasInstance {
			.blas = scene.blases[instances[i].meshIndex],
			.transform = instances[i].transform,
			.instanceCustomIndex = instances[i].customIndex,
			.rayMask = instances[i].rayMask,
		};
	}
	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::compute }
	);
	vkof::render_node_callback({
		.node = node,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			vkof::tlas_build(
				cmd,
				scene.tlas,
				srat::slice<vkof::TlasInstance const>(
					tlasInstances.data(), tlasInstances.size()
				)
			);
		},
	});
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(&node, 1u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(node);
	gpu_wait();

	return scene;
}

inline void geometry_scene_destroy(GeometryScene const & scene) {
	vkof::tlas_destroy(scene.tlas);
	for (auto const & blas : scene.blases) { vkof::blas_destroy(blas); }
	for (auto const & buf : scene.indexBuffers) { vkof::buffer_destroy(buf); }
	for (auto const & buf : scene.vertexBuffers) { vkof::buffer_destroy(buf); }
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
