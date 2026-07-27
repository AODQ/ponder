#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <ponder/energy-tables.hpp>
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// scene-level path-tracer validation renders, built on the cornell box from
// test-cornell.cpp. three classic scenes exercise the parts of the openpbr
// transmission chain no earlier test reaches: nested dielectric media
// (relative-ior interfaces via a medium stack), hollow shells, and a
// many-instance tlas. each test writes a high-resolution png for the
// eyeball pass and asserts physical invariants the images must satisfy:
//
// - nested spheres: an inner dielectric whose ior matches the outer glass
//   must be invisible (render equals the solid-sphere render), while a
//   mismatched inner ior must produce a measurably different image
// - hollow glass sphere vs checkerboard: an ior-1 shell must be invisible
//   (render equals the sphere-less background), and the ray through the
//   sphere center passes undeviated at every ior -- the checker cell it
//   lands on keeps its color, dimmed by no more than the analytic
//   normal-incidence fresnel transmittance of the 4 interfaces
// - bouncing sphere field: thousands of instanced spheres over a plane
//   under a sky; asserts instance-index integrity across a large tlas
//
// all renders share dielectric_scene_render.comp, a bsdf-sampling path
// tracer through the full ported openPbrSampleWo / openPbrEvaluateF chain
// that -- unlike cornell_openpbr_render.comp -- follows transmitted rays
// across medium boundaries, tracking nested media as a stack of absolute
// iors and presenting each boundary to the material api as its exact
// relative-ior equivalent
// ---------------------------------------------------------------------------

namespace {

struct Vertex { f32 x, y, z; };

// cornell walls, same geometry as test-cornell.cpp: interior box [-1,1]^3,
// front face (z=-1) open, camera outside at (0,0,-3) looking down +z.
// wall quads extend to +-1.05 so adjacent walls overlap at the corners
// (watertightness holds only within a blas, see test-cornell.cpp)
static u32 const skWallFloor = 0u;
static u32 const skWallCeiling = 1u;
static u32 const skWallBack = 2u;
static u32 const skWallLeft = 3u;
static u32 const skWallRight = 4u;
static u32 const skWallLight = 5u;
static u32 const skWallCount = 6u;

static f32 const skE = 1.05f;

static Vertex const skWallQuads[skWallCount][4] = {
	// floor (y=-1)
	{
		{ -skE, -1.0f, -skE },
		{ skE, -1.0f, -skE },
		{ skE, -1.0f, skE },
		{ -skE, -1.0f, skE },
	},
	// ceiling (y=+1)
	{
		{ -skE, 1.0f, -skE },
		{ -skE, 1.0f, skE },
		{ skE, 1.0f, skE },
		{ skE, 1.0f, -skE },
	},
	// back wall (z=+1)
	{
		{ -skE, -skE, 1.0f },
		{ skE, -skE, 1.0f },
		{ skE, skE, 1.0f },
		{ -skE, skE, 1.0f },
	},
	// left wall (x=-1)
	{
		{ -1.0f, -skE, -skE },
		{ -1.0f, -skE, skE },
		{ -1.0f, skE, skE },
		{ -1.0f, skE, -skE },
	},
	// right wall (x=+1)
	{
		{ 1.0f, -skE, -skE },
		{ 1.0f, skE, -skE },
		{ 1.0f, skE, skE },
		{ 1.0f, -skE, skE },
	},
	// area light slightly below the ceiling, wound to emit downward only
	// (see the radiosity-cavity note in test-cornell.cpp)
	{
		{ -0.25f, 0.998f, -0.25f },
		{ 0.25f, 0.998f, -0.25f },
		{ 0.25f, 0.998f, 0.25f },
		{ -0.25f, 0.998f, 0.25f },
	},
};

static u32 const skQuadIndices[6] = { 0u, 1u, 2u, 0u, 2u, 3u };

static f32m44 const skIdentity = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
};

// uniform scale + translation (f32m44 is m[col*4+row]); spheres use
// analytic normals in the shader so scale may live in the tlas transform
static f32m44 transform_scale_translate(
	f32 const scale, f32 const x, f32 const y, f32 const z
) {
	return f32m44 {
		scale, 0.0f, 0.0f, 0.0f,
		0.0f, scale, 0.0f, 0.0f,
		0.0f, 0.0f, scale, 0.0f,
		x, y, z, 1.0f,
	};
}

// -- uv sphere mesh, watertight (shared ring vertices, single pole verts)

struct MeshData {
	std::vector<Vertex> vertices;
	std::vector<u32> indices;
};

static MeshData sphere_mesh(u32 const rings, u32 const segments) {
	MeshData mesh;
	f32 const pi = 3.14159265358979323846f;
	mesh.vertices.push_back(Vertex { 0.0f, 1.0f, 0.0f });
	for (u32 r = 1u; r < rings; ++r) {
		f32 const theta = pi * static_cast<f32>(r) / static_cast<f32>(rings);
		for (u32 s = 0u; s < segments; ++s) {
			f32 const phi = (
				2.0f * pi * static_cast<f32>(s) / static_cast<f32>(segments)
			);
			mesh.vertices.push_back(Vertex {
				std::sin(theta) * std::cos(phi),
				std::cos(theta),
				std::sin(theta) * std::sin(phi),
			});
		}
	}
	u32 const bottomPole = static_cast<u32>(mesh.vertices.size());
	mesh.vertices.push_back(Vertex { 0.0f, -1.0f, 0.0f });

	auto ringVertex = [&](u32 const ring, u32 const segment) {
		return 1u + ring * segments + (segment % segments);
	};
	for (u32 s = 0u; s < segments; ++s) {
		mesh.indices.push_back(0u);
		mesh.indices.push_back(ringVertex(0u, s + 1u));
		mesh.indices.push_back(ringVertex(0u, s));
	}
	for (u32 r = 1u; r < rings - 1u; ++r) {
		for (u32 s = 0u; s < segments; ++s) {
			u32 const a = ringVertex(r - 1u, s);
			u32 const b = ringVertex(r - 1u, s + 1u);
			u32 const c = ringVertex(r, s + 1u);
			u32 const d = ringVertex(r, s);
			mesh.indices.push_back(a);
			mesh.indices.push_back(b);
			mesh.indices.push_back(c);
			mesh.indices.push_back(a);
			mesh.indices.push_back(c);
			mesh.indices.push_back(d);
		}
	}
	for (u32 s = 0u; s < segments; ++s) {
		mesh.indices.push_back(ringVertex(rings - 2u, s));
		mesh.indices.push_back(ringVertex(rings - 2u, s + 1u));
		mesh.indices.push_back(bottomPole);
	}
	return mesh;
}

// -- scene builder: meshes uploaded once, instanced into a fresh tlas

static vkof::Buffer upload_bytes(void const * data, u64 const byteCount) {
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

struct InstanceDef {
	u32 meshIndex;
	f32m44 transform;
	u32 customIndex;
};

struct Scene {
	std::vector<vkof::Buffer> vertexBuffers;
	std::vector<vkof::Buffer> indexBuffers;
	std::vector<vkof::AccelerationStructureBlas> blases;
	vkof::AccelerationStructureTlas tlas;
};

static Scene scene_create(
	std::vector<MeshData> const & meshes,
	std::vector<InstanceDef> const & instances
) {
	Scene scene {};
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
			.positionVa = vkof::buffer_virtual_address(
				scene.vertexBuffers[m]
			),
			.vertexCount = static_cast<u32>(meshes[m].vertices.size()),
			.indexVa = vkof::buffer_virtual_address(scene.indexBuffers[m]),
			.triangleCount = static_cast<u32>(meshes[m].indices.size() / 3u),
		});
		REQUIRE(scene.blases[m].id != 0);
	}

	scene.tlas = vkof::tlas_create({
		.maxInstances = static_cast<u32>(instances.size()),
	});
	REQUIRE(scene.tlas.id != 0);

	std::vector<vkof::TlasInstance> tlasInstances(instances.size());
	for (u32 i = 0u; i < instances.size(); ++i) {
		tlasInstances[i] = vkof::TlasInstance {
			.blas = scene.blases[instances[i].meshIndex],
			.transform = instances[i].transform,
			.instanceCustomIndex = instances[i].customIndex,
			.rayMask = 0xFFu,
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
	test::gpu_wait();

	return scene;
}

static void scene_destroy(Scene const & scene) {
	vkof::tlas_destroy(scene.tlas);
	for (auto const & blas : scene.blases) { vkof::blas_destroy(blas); }
	for (auto const & buf : scene.indexBuffers) { vkof::buffer_destroy(buf); }
	for (auto const & buf : scene.vertexBuffers) {
		vkof::buffer_destroy(buf);
	}
}

// -- instance records, mirrors InstanceRecord in dielectric_scene_render.comp

static u32 const skFlagSphere = 1u;
static u32 const skFlagChecker = 2u;

struct InstanceRecord {
	u64 vertexVa;
	u64 indexVa;
	f32 baseColor[3];
	f32 specularWeight;
	f32 emission[3];
	f32 specularRoughness;
	f32 transmissionColor[3];
	f32 transmissionWeight;
	f32 ior;
	u32 flags;
	f32 baseMetalness;
	f32 pad;
};
static_assert(sizeof(InstanceRecord) == 80u);

static InstanceRecord record_diffuse(
	f32 const r, f32 const g, f32 const b,
	f32 const emission = 0.0f,
	u32 const flags = 0u
) {
	return InstanceRecord {
		.vertexVa = 0u,
		.indexVa = 0u,
		.baseColor = { r, g, b },
		.specularWeight = 0.0f,
		.emission = { emission, emission, emission },
		.specularRoughness = 0.3f,
		.transmissionColor = { 1.0f, 1.0f, 1.0f },
		.transmissionWeight = 0.0f,
		.ior = 1.5f,
		.flags = flags,
		.baseMetalness = 0.0f,
		.pad = 0.0f,
	};
}

// smooth solid dielectric: full transmission, ior of the enclosed medium
static InstanceRecord record_glass(f32 const interiorIor) {
	return InstanceRecord {
		.vertexVa = 0u,
		.indexVa = 0u,
		.baseColor = { 1.0f, 1.0f, 1.0f },
		.specularWeight = 1.0f,
		.emission = { 0.0f, 0.0f, 0.0f },
		.specularRoughness = 0.02f,
		.transmissionColor = { 1.0f, 1.0f, 1.0f },
		.transmissionWeight = 1.0f,
		.ior = interiorIor,
		.flags = skFlagSphere,
		.baseMetalness = 0.0f,
		.pad = 0.0f,
	};
}

// -- camera + render plumbing

struct Camera {
	f32 origin[3];
	f32 right[3];
	f32 up[3];
	f32 forward[3];
	f32 tanHalfFovX;
	f32 tanHalfFovY;
};

// the cornell-box camera every wall-scene test uses; square pixels
// (tanY = tanX * h / w at 4:3)
static Camera const skCornellCamera {
	.origin = { 0.0f, 0.0f, -3.0f },
	.right = { 1.0f, 0.0f, 0.0f },
	.up = { 0.0f, 1.0f, 0.0f },
	.forward = { 0.0f, 0.0f, 1.0f },
	.tanHalfFovX = 0.5f,
	.tanHalfFovY = 0.375f,
};

static Camera camera_look_at(
	f32 const ox, f32 const oy, f32 const oz,
	f32 const tx, f32 const ty, f32 const tz,
	f32 const tanHalfFovX, f32 const tanHalfFovY
) {
	f32 fwd[3] = { tx - ox, ty - oy, tz - oz };
	f32 const fl = std::sqrt(
		fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]
	);
	fwd[0] /= fl; fwd[1] /= fl; fwd[2] /= fl;
	// right = normalize(worldUp x fwd), up = fwd x right
	f32 right[3] = { fwd[2], 0.0f, -fwd[0] };
	f32 const rl = std::sqrt(right[0] * right[0] + right[2] * right[2]);
	right[0] /= rl; right[2] /= rl;
	f32 const up[3] = {
		fwd[1] * right[2] - fwd[2] * right[1],
		fwd[2] * right[0] - fwd[0] * right[2],
		fwd[0] * right[1] - fwd[1] * right[0],
	};
	return Camera {
		.origin = { ox, oy, oz },
		.right = { right[0], right[1], right[2] },
		.up = { up[0], up[1], up[2] },
		.forward = { fwd[0], fwd[1], fwd[2] },
		.tanHalfFovX = tanHalfFovX,
		.tanHalfFovY = tanHalfFovY,
	};
}

// cpu double-precision replica of the shader's primary-ray math, for the
// analytic assertions (disk pixel counts, checker target points)
struct Vec3d { f64 x, y, z; };

static Vec3d cpu_primary_dir(
	Camera const & cam, u32 const width, u32 const height,
	u32 const px, u32 const py
) {
	f64 const ndcX = (
		2.0 * (static_cast<f64>(px) + 0.5) / static_cast<f64>(width) - 1.0
	);
	f64 const ndcY = (
		2.0 * (static_cast<f64>(py) + 0.5) / static_cast<f64>(height) - 1.0
	);
	f64 const sx = ndcX * cam.tanHalfFovX;
	f64 const sy = -ndcY * cam.tanHalfFovY;
	Vec3d d {
		cam.forward[0] + cam.right[0] * sx + cam.up[0] * sy,
		cam.forward[1] + cam.right[1] * sx + cam.up[1] * sy,
		cam.forward[2] + cam.right[2] * sx + cam.up[2] * sy,
	};
	f64 const l = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
	return Vec3d { d.x / l, d.y / l, d.z / l };
}

// how many pixels' primary rays hit a perfect sphere, in f64; the gpu
// count (against the tessellated mesh) must agree to within the thin
// silhouette band the tessellation error can flip
static u32 cpu_sphere_disk_pixels(
	Camera const & cam, u32 const width, u32 const height,
	f64 const cx, f64 const cy, f64 const cz, f64 const radius
) {
	u32 count = 0u;
	for (u32 py = 0u; py < height; ++py) {
		for (u32 px = 0u; px < width; ++px) {
			Vec3d const d = cpu_primary_dir(cam, width, height, px, py);
			f64 const ocx = cam.origin[0] - cx;
			f64 const ocy = cam.origin[1] - cy;
			f64 const ocz = cam.origin[2] - cz;
			f64 const b = ocx * d.x + ocy * d.y + ocz * d.z;
			f64 const c = (
				ocx * ocx + ocy * ocy + ocz * ocz - radius * radius
			);
			f64 const disc = b * b - c;
			if (disc >= 0.0 && (-b - std::sqrt(disc)) > 0.0) {
				count++;
			}
		}
	}
	return count;
}

// must match checker_color in dielectric_scene_render.comp
static bool checker_is_bright(f64 const x, f64 const y, f64 const z) {
	i64 const parity = (
		static_cast<i64>(std::floor(x * 4.0 + 0.5))
		+ static_cast<i64>(std::floor(y * 4.0 + 0.5))
		+ static_cast<i64>(std::floor(z * 4.0 + 0.5))
	);
	return (parity & 1) == 0;
}

// mirrors PC in dielectric_scene_render.comp (scalar packing)
struct Push {
	u64 outVa;
	u64 sceneVa;
	u64 primaryHitVa;
	u32 width;
	u32 height;
	u32 sampleOffset;
	u32 sampleCount;
	u32 bounceCount;
	u32 kullaContyHandle;
	u32 skyEnable;
	f32 camOrigin[3];
	f32 camRight[3];
	f32 camUp[3];
	f32 camForward[3];
	f32 tanHalfFovX;
	f32 tanHalfFovY;
};
static_assert(sizeof(Push) == 112u);

struct RenderParams {
	vkof::Pipeline pipeline;
	std::vector<InstanceRecord> const * records;
	Camera camera;
	u32 width;
	u32 height;
	u32 sampleCount;
	u32 bounceCount;
	bool sky;
	u32 kullaContyHandle;
};

struct RenderResult {
	// planar rgb, radiance averaged over all samples
	std::vector<f32> rgb;
	// per-pixel primary-hit instanceCustomIndex, 0xFFFFFFFF on miss
	std::vector<u32> primaryHit;
};

// dispatches the render in sample chunks (each chunk is a full-frame
// dispatch accumulating into the output buffer) so no single dispatch
// runs long enough to trip a gpu watchdog
static RenderResult render_scene(RenderParams const & params) {
	u32 const pixelCount = params.width * params.height;
	u32 const skChunk = 32u;

	auto outBuf = vkof::buffer_create({
		.byteCount = static_cast<u64>(3u * pixelCount) * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	auto hitBuf = test::make_buffer_u32(pixelCount);
	auto sceneBuf = upload_bytes(
		params.records->data(),
		params.records->size() * sizeof(InstanceRecord)
	);

	for (u32 offset = 0u; offset < params.sampleCount; offset += skChunk) {
		u32 const chunk = std::min(skChunk, params.sampleCount - offset);
		Push const push {
			.outVa = vkof::buffer_virtual_address(outBuf),
			.sceneVa = vkof::buffer_virtual_address(sceneBuf),
			.primaryHitVa = vkof::buffer_virtual_address(hitBuf),
			.width = params.width,
			.height = params.height,
			.sampleOffset = offset,
			.sampleCount = chunk,
			.bounceCount = params.bounceCount,
			.kullaContyHandle = params.kullaContyHandle,
			.skyEnable = params.sky ? 1u : 0u,
			.camOrigin = {
				params.camera.origin[0],
				params.camera.origin[1],
				params.camera.origin[2],
			},
			.camRight = {
				params.camera.right[0],
				params.camera.right[1],
				params.camera.right[2],
			},
			.camUp = {
				params.camera.up[0],
				params.camera.up[1],
				params.camera.up[2],
			},
			.camForward = {
				params.camera.forward[0],
				params.camera.forward[1],
				params.camera.forward[2],
			},
			.tanHalfFovX = params.camera.tanHalfFovX,
			.tanHalfFovY = params.camera.tanHalfFovY,
		};
		test::dispatch(
			params.pipeline,
			push,
			params.width / 8u,
			params.height / 8u
		);
		test::gpu_wait();
	}

	RenderResult result;
	result.rgb = test::readback<f32>(outBuf, 0u, 3u * pixelCount);
	result.primaryHit = test::readback<u32>(hitBuf, 0u, pixelCount);
	for (auto & v : result.rgb) {
		v /= static_cast<f32>(params.sampleCount);
	}

	vkof::buffer_destroy(sceneBuf);
	vkof::buffer_destroy(hitBuf);
	vkof::buffer_destroy(outBuf);
	return result;
}

// -- image analysis helpers

struct ImageStats {
	u32 nanCount;
	f32 minValue;
	f32 maxValue;
	f64 mean;
};

static ImageStats image_stats(std::vector<f32> const & rgb) {
	ImageStats stats { 0u, 0.0f, 0.0f, 0.0 };
	f64 sum = 0.0;
	for (f32 const v : rgb) {
		if (!std::isfinite(v)) {
			stats.nanCount++;
			continue;
		}
		stats.minValue = std::min(stats.minValue, v);
		stats.maxValue = std::max(stats.maxValue, v);
		sum += v;
	}
	stats.mean = sum / static_cast<f64>(rgb.size());
	return stats;
}

// per-block mean radiance (all channels pooled) on a blockSize grid;
// noise-robust unit for comparing two renders of equivalent scenes
static std::vector<f64> block_means(
	std::vector<f32> const & rgb,
	u32 const width, u32 const height, u32 const blockSize
) {
	u32 const pixelCount = width * height;
	u32 const bw = width / blockSize;
	u32 const bh = height / blockSize;
	std::vector<f64> means(bw * bh, 0.0);
	for (u32 py = 0u; py < bh * blockSize; ++py) {
		for (u32 px = 0u; px < bw * blockSize; ++px) {
			u32 const i = py * width + px;
			u32 const b = (py / blockSize) * bw + (px / blockSize);
			means[b] += (
				static_cast<f64>(rgb[i])
				+ rgb[pixelCount + i]
				+ rgb[2u * pixelCount + i]
			);
		}
	}
	for (auto & m : means) {
		m /= 3.0 * blockSize * blockSize;
	}
	return means;
}

struct BlockDiff {
	f64 mean;
	f64 max;
};

static BlockDiff block_diff(
	std::vector<f64> const & a, std::vector<f64> const & b
) {
	BlockDiff diff { 0.0, 0.0 };
	for (u32 i = 0u; i < a.size(); ++i) {
		f64 const rel = (
			std::fabs(a[i] - b[i]) / std::max(std::max(a[i], b[i]), 0.02)
		);
		diff.mean += rel;
		diff.max = std::max(diff.max, rel);
	}
	diff.mean /= static_cast<f64>(a.size());
	return diff;
}

// mean luminance of a (2*halfExtent+1)^2 pixel patch
static f64 patch_luminance(
	std::vector<f32> const & rgb,
	u32 const width, u32 const height,
	u32 const cx, u32 const cy, u32 const halfExtent
) {
	u32 const pixelCount = width * height;
	f64 sum = 0.0;
	u32 count = 0u;
	for (u32 py = cy - halfExtent; py <= cy + halfExtent; ++py) {
		for (u32 px = cx - halfExtent; px <= cx + halfExtent; ++px) {
			u32 const i = py * width + px;
			sum += (
				0.2126 * rgb[i]
				+ 0.7152 * rgb[pixelCount + i]
				+ 0.0722 * rgb[2u * pixelCount + i]
			);
			count++;
		}
	}
	return sum / count;
}

static void save_png(
	std::vector<f32> const & rgb,
	u32 const width, u32 const height,
	std::string const & path
) {
	u32 const pixelCount = width * height;
	std::vector<f32> r(pixelCount), g(pixelCount), b(pixelCount);
	for (u32 i = 0u; i < pixelCount; ++i) {
		r[i] = std::pow(std::clamp(rgb[i], 0.0f, 1.0f), 1.0f / 2.2f);
		g[i] = std::pow(
			std::clamp(rgb[pixelCount + i], 0.0f, 1.0f), 1.0f / 2.2f
		);
		b[i] = std::pow(
			std::clamp(rgb[2u * pixelCount + i], 0.0f, 1.0f), 1.0f / 2.2f
		);
	}
	CHECK(test::write_heatmap_png(
		r, g, b, width, height, path.c_str()
	));
}

// analytic normal-incidence transmittance through the 4 interfaces of a
// concentric shell (or 4-surface nested pair) along the center ray; the
// straight-through path alone carries at least this fraction, and
// internal re-reflections only add on top
static f64 normal_incidence_t4(f64 const ior) {
	f64 const f0 = ((ior - 1.0) / (ior + 1.0)) * ((ior - 1.0) / (ior + 1.0));
	return std::pow(1.0 - f0, 4.0);
}

// wall records shared by the two cornell-based tests; caller appends the
// sphere records. light emission matches cornell_render.comp's 15
static f32 const skLe = 15.0f;

static std::vector<InstanceRecord> cornell_wall_records(
	Scene const & scene,
	bool const checkerBack
) {
	std::vector<InstanceRecord> records(skWallCount);
	for (u32 wall = 0u; wall < skWallCount; ++wall) {
		if (checkerBack) {
			// neutral walls so the checkerboard image through the glass
			// stays uncontaminated by wall color bleed
			records[wall] = record_diffuse(0.60f, 0.60f, 0.60f);
		} else {
			records[wall] = record_diffuse(0.73f, 0.73f, 0.73f);
		}
	}
	if (checkerBack) {
		records[skWallFloor] = record_diffuse(0.73f, 0.73f, 0.73f);
		records[skWallCeiling] = record_diffuse(0.73f, 0.73f, 0.73f);
		records[skWallBack] = record_diffuse(
			0.85f, 0.85f, 0.85f, 0.0f, skFlagChecker
		);
	} else {
		records[skWallLeft] = record_diffuse(0.63f, 0.065f, 0.05f);
		records[skWallRight] = record_diffuse(0.14f, 0.45f, 0.091f);
	}
	records[skWallLight] = record_diffuse(0.78f, 0.78f, 0.78f, skLe);
	for (u32 wall = 0u; wall < skWallCount; ++wall) {
		records[wall].vertexVa = vkof::buffer_virtual_address(
			scene.vertexBuffers[wall]
		);
		records[wall].indexVa = vkof::buffer_virtual_address(
			scene.indexBuffers[wall]
		);
	}
	return records;
}

static std::vector<MeshData> cornell_meshes_with_sphere() {
	std::vector<MeshData> meshes(skWallCount);
	for (u32 wall = 0u; wall < skWallCount; ++wall) {
		meshes[wall].vertices.assign(
			skWallQuads[wall], skWallQuads[wall] + 4
		);
		meshes[wall].indices.assign(skQuadIndices, skQuadIndices + 6);
	}
	// hero sphere: dense tessellation so the silhouette band where the
	// mesh disagrees with the analytic sphere stays under ~0.1% of the
	// radius (the disk-pixel-count assertions budget for this)
	meshes.push_back(sphere_mesh(64u, 96u));
	return meshes;
}

static u32 const skSphereMesh = skWallCount;

// shared render size for the two cornell-based tests
static u32 const skWidth = 512u;
static u32 const skHeight = 384u;
static u32 const skSampleCount = 384u;
static u32 const skBounceCount = 16u;

}

TEST_SUITE("[headless]") {

// ---------------------------------------------------------------------------
// scene 1: a dielectric sphere nested inside a transmission dielectric
// sphere, swept over inner/outer ior combinations. the load-bearing
// physical assertion is the matched-ior pair: when the inner medium's ior
// equals the outer's, the inner interface has relative ior 1 and must be
// perfectly invisible -- the render must match the solid single-sphere
// render. a mismatched inner ior must measurably change the image
// ---------------------------------------------------------------------------

TEST_CASE("dielectric: nested spheres ior sweep") {
	auto energyTables = ponder::energy_tables_create();
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "dielectric_scene_render.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto const meshes = cornell_meshes_with_sphere();

	f32 const outerRadius = 0.45f;
	f32 const innerRadius = 0.22f;
	f32 const centerY = -0.35f;

	struct Config {
		char const * name;
		bool hasInner;
		f32 outerIor;
		f32 innerIor;
	};
	Config const configs[] = {
		{ "solid-glass", false, 1.5f, 0.0f },
		{ "matched-inner", true, 1.5f, 1.5f },
		{ "water-in-glass", true, 1.5f, 1.33f },
		{ "diamond-in-glass", true, 1.5f, 2.42f },
		{ "glass-in-water", true, 1.33f, 1.5f },
	};

	u32 const skOuterInstance = skWallCount;
	u32 const skInnerInstance = skWallCount + 1u;

	u32 const cpuDiskPixels = cpu_sphere_disk_pixels(
		skCornellCamera, skWidth, skHeight,
		0.0, centerY, 0.0, outerRadius
	);
	REQUIRE(cpuDiskPixels > 5000u);

	std::vector<f64> solidBlocks;
	std::vector<f64> matchedBlocks;
	std::vector<f64> diamondBlocks;

	for (auto const & config : configs) {
		std::string const configName(config.name);
		CAPTURE(configName);

		std::vector<InstanceDef> instances;
		for (u32 wall = 0u; wall < skWallCount; ++wall) {
			instances.push_back({ wall, skIdentity, wall });
		}
		instances.push_back({
			skSphereMesh,
			transform_scale_translate(outerRadius, 0.0f, centerY, 0.0f),
			skOuterInstance,
		});
		if (config.hasInner) {
			instances.push_back({
				skSphereMesh,
				transform_scale_translate(innerRadius, 0.0f, centerY, 0.0f),
				skInnerInstance,
			});
		}
		auto const scene = scene_create(meshes, instances);
		vkof::acceleration_structure_set_tlas(scene.tlas);

		auto records = cornell_wall_records(scene, /*checkerBack=*/false);
		records.push_back(record_glass(config.outerIor));
		if (config.hasInner) {
			records.push_back(record_glass(config.innerIor));
		}

		auto const result = render_scene({
			.pipeline = pl,
			.records = &records,
			.camera = skCornellCamera,
			.width = skWidth,
			.height = skHeight,
			.sampleCount = skSampleCount,
			.bounceCount = skBounceCount,
			.sky = false,
			.kullaContyHandle = energyTables.kullaContyEnergyHandle,
		});

		auto const stats = image_stats(result.rgb);
		CHECK(stats.nanCount == 0u);
		CHECK(stats.minValue >= 0.0f);
		CHECK(stats.mean > 0.003);
		CHECK(stats.mean < 2.0);

		// primary rays against the sphere silhouette: the gpu disk pixel
		// count must match the f64 perfect-sphere count, with a small
		// budget for the tessellated silhouette band and shared edges
		u32 gpuDiskPixels = 0u;
		u32 innerPrimaryHits = 0u;
		for (u32 const inst : result.primaryHit) {
			if (inst == skOuterInstance) { gpuDiskPixels++; }
			if (inst == skInnerInstance) { innerPrimaryHits++; }
		}
		CHECK(
			std::abs(
				static_cast<i64>(gpuDiskPixels)
				- static_cast<i64>(cpuDiskPixels)
			)
			< static_cast<i64>(cpuDiskPixels / 50u + 32u)
		);
		// the inner sphere is entirely enclosed by the outer: no primary
		// ray may ever report it as the first hit
		CHECK(innerPrimaryHits == 0u);

		auto const blocks = block_means(result.rgb, skWidth, skHeight, 32u);
		if (configName == "solid-glass") { solidBlocks = blocks; }
		if (configName == "matched-inner") { matchedBlocks = blocks; }
		if (configName == "diamond-in-glass") { diamondBlocks = blocks; }

		save_png(
			result.rgb, skWidth, skHeight,
			std::string(DIELECTRIC_OUTPUT_DIR "dielectric-nested-")
			+ config.name + ".png"
		);
	}

	// matched inner ior: interface relative ior 1 -> delta passthrough ->
	// image identical to the solid sphere up to monte-carlo noise
	BlockDiff const matchedDiff = block_diff(solidBlocks, matchedBlocks);
	CHECK(matchedDiff.mean < 0.05);
	CHECK(matchedDiff.max < 0.35);

	// teeth: a genuinely different inner ior must move the image by a
	// clear margin over the matched-pair noise floor
	BlockDiff const diamondDiff = block_diff(solidBlocks, diamondBlocks);
	CHECK(diamondDiff.mean > 2.0 * matchedDiff.mean);

	vkof::pipeline_destroy(pl);
	ponder::energy_tables_destroy(energyTables);
}

// ---------------------------------------------------------------------------
// scene 2: a hollow glass sphere (glass shell, air inside) in front of a
// checkerboard back wall, swept over shell ior. two physical anchors:
// an ior-1 shell is invisible (render equals the sphere-less background),
// and the ray through the exact sphere center crosses all 4 interfaces at
// normal incidence, passing undeviated at every ior -- the checker cell it
// lands on keeps its color, dimmed by no more than the analytic fresnel
// transmittance (internal re-reflections only add radiance back)
// ---------------------------------------------------------------------------

TEST_CASE("dielectric: hollow glass sphere vs checkerboard") {
	auto energyTables = ponder::energy_tables_create();
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "dielectric_scene_render.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	auto const meshes = cornell_meshes_with_sphere();

	f32 const outerRadius = 0.5f;
	f32 const innerRadius = 0.44f;
	// sphere center chosen so the ray through pixel (256, 256) passes
	// exactly through it: ndcY(py=256) = 0.3359375, so
	// centerY = -3 * 0.3359375 * tanHalfFovY = -0.377930. that ray then
	// continues undeviated to the back wall at y = centerY * 4/3, which
	// must land mid-cell in a bright checker cell (REQUIREd below)
	f32 const centerY = -0.377930f;
	u32 const targetPx = 256u;
	u32 const targetPy = 256u;

	// verify the design invariant in f64 with the exact camera math
	{
		Vec3d const d = cpu_primary_dir(
			skCornellCamera, skWidth, skHeight, targetPx, targetPy
		);
		f64 const s = (1.0 - (-3.0)) / d.z;
		f64 const wx = 0.0 + s * d.x;
		f64 const wy = 0.0 + s * d.y;
		REQUIRE(checker_is_bright(wx, wy, 1.0));
		// and the ray indeed passes through the sphere center: closest
		// approach to (0, centerY, 0) is well under a thousandth of the
		// radius, so refraction deviation stays negligible
		f64 const t0 = 3.0 / d.z;
		f64 const missX = t0 * d.x;
		f64 const missY = -3.0 * 0.0 + t0 * d.y - centerY;
		REQUIRE(std::sqrt(missX * missX + missY * missY) < 0.01);
	}

	u32 const skOuterInstance = skWallCount;
	u32 const skInnerInstance = skWallCount + 1u;

	struct Config {
		char const * name;
		bool hasSphere;
		f32 ior;
	};
	Config const configs[] = {
		{ "background", false, 0.0f },
		{ "ior-1.0", true, 1.0f },
		{ "ior-1.33", true, 1.33f },
		{ "ior-1.5", true, 1.5f },
		{ "ior-2.42", true, 2.42f },
	};

	u32 const cpuDiskPixels = cpu_sphere_disk_pixels(
		skCornellCamera, skWidth, skHeight,
		0.0, centerY, 0.0, outerRadius
	);
	REQUIRE(cpuDiskPixels > 5000u);

	f64 backgroundPatch = 0.0;
	std::vector<f64> backgroundBlocks;
	std::vector<f64> ior1Blocks;
	std::vector<f64> ior15Blocks;

	for (auto const & config : configs) {
		std::string const configName(config.name);
		CAPTURE(configName);

		std::vector<InstanceDef> instances;
		for (u32 wall = 0u; wall < skWallCount; ++wall) {
			instances.push_back({ wall, skIdentity, wall });
		}
		if (config.hasSphere) {
			instances.push_back({
				skSphereMesh,
				transform_scale_translate(outerRadius, 0.0f, centerY, 0.0f),
				skOuterInstance,
			});
			instances.push_back({
				skSphereMesh,
				transform_scale_translate(innerRadius, 0.0f, centerY, 0.0f),
				skInnerInstance,
			});
		}
		auto const scene = scene_create(meshes, instances);
		vkof::acceleration_structure_set_tlas(scene.tlas);

		auto records = cornell_wall_records(scene, /*checkerBack=*/true);
		if (config.hasSphere) {
			// hollow shell: the outer surface enters glass, the inner
			// surface encloses plain air again
			records.push_back(record_glass(config.ior));
			records.push_back(record_glass(1.0f));
		}

		auto const result = render_scene({
			.pipeline = pl,
			.records = &records,
			.camera = skCornellCamera,
			.width = skWidth,
			.height = skHeight,
			.sampleCount = skSampleCount,
			.bounceCount = skBounceCount,
			.sky = false,
			.kullaContyHandle = energyTables.kullaContyEnergyHandle,
		});

		auto const stats = image_stats(result.rgb);
		CHECK(stats.nanCount == 0u);
		CHECK(stats.minValue >= 0.0f);
		CHECK(stats.mean > 0.003);
		CHECK(stats.mean < 2.0);

		if (config.hasSphere) {
			u32 gpuDiskPixels = 0u;
			u32 innerPrimaryHits = 0u;
			for (u32 const inst : result.primaryHit) {
				if (inst == skOuterInstance) { gpuDiskPixels++; }
				if (inst == skInnerInstance) { innerPrimaryHits++; }
			}
			CHECK(
				std::abs(
					static_cast<i64>(gpuDiskPixels)
					- static_cast<i64>(cpuDiskPixels)
				)
				< static_cast<i64>(cpuDiskPixels / 50u + 32u)
			);
			CHECK(innerPrimaryHits == 0u);
		}

		f64 const patch = patch_luminance(
			result.rgb, skWidth, skHeight, targetPx, targetPy, 2u
		);
		auto const blocks = block_means(result.rgb, skWidth, skHeight, 32u);

		if (configName == "background") {
			backgroundPatch = patch;
			backgroundBlocks = blocks;
			// the target cell is bright and lit
			CHECK(patch > 0.02);
		} else {
			// undeviated center ray: the checker cell color survives,
			// dimmed at most by the straight-through fresnel product;
			// re-reflected paths can only add radiance back on top
			f64 const t4 = normal_incidence_t4(config.ior);
			f64 const ratio = patch / backgroundPatch;
			CAPTURE(ratio);
			CAPTURE(t4);
			CHECK(ratio > 0.65 * t4);
			CHECK(ratio < 1.15);
		}

		if (configName == "ior-1.0") { ior1Blocks = blocks; }
		if (configName == "ior-1.5") { ior15Blocks = blocks; }

		save_png(
			result.rgb, skWidth, skHeight,
			std::string(DIELECTRIC_OUTPUT_DIR "dielectric-hollow-")
			+ config.name + ".png"
		);
	}

	// an ior-1 shell has relative ior 1 at all 4 interfaces: perfectly
	// invisible, so the render must equal the sphere-less background
	BlockDiff const invisibleDiff = block_diff(backgroundBlocks, ior1Blocks);
	CHECK(invisibleDiff.mean < 0.05);
	CHECK(invisibleDiff.max < 0.35);

	// teeth: real glass must visibly bend the checkerboard
	BlockDiff const glassDiff = block_diff(backgroundBlocks, ior15Blocks);
	CHECK(glassDiff.mean > 2.0 * invisibleDiff.mean);

	vkof::pipeline_destroy(pl);
	ponder::energy_tables_destroy(energyTables);
}

// ---------------------------------------------------------------------------
// scene 3: a flat plane with thousands of instanced spheres frozen
// mid-bounce (heights follow a traveling |sin| wave), lit by a gradient
// sky. exercises instance-index integrity and material fan-out across a
// large tlas: over a thousand distinct instances must be identifiable
// from primary hits alone, every reported index must be valid, and the
// image must contain plane, spheres, and sky simultaneously
// ---------------------------------------------------------------------------

TEST_CASE("dielectric: bouncing sphere field") {
	auto energyTables = ponder::energy_tables_create();
	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "dielectric_scene_render.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	u32 const width = 896u;
	u32 const height = 504u;
	u32 const sampleCount = 128u;
	u32 const bounceCount = 4u;

	u32 const gridX = 56u;
	u32 const gridZ = 56u;
	u32 const sphereCount = gridX * gridZ;
	f32 const radius = 0.25f;
	f32 const spacing = 0.62f;
	f32 const bounceHeight = 1.8f;

	// meshes: ground plane + a coarse sphere (analytic normals carry the
	// shading; the silhouettes are a few dozen pixels at most)
	std::vector<MeshData> meshes(2u);
	meshes[0].vertices = {
		{ -60.0f, 0.0f, -60.0f },
		{ 60.0f, 0.0f, -60.0f },
		{ 60.0f, 0.0f, 60.0f },
		{ -60.0f, 0.0f, 60.0f },
	};
	meshes[0].indices.assign(skQuadIndices, skQuadIndices + 6);
	meshes[1] = sphere_mesh(16u, 24u);

	std::vector<InstanceDef> instances;
	instances.push_back({ 0u, skIdentity, 0u });

	std::vector<InstanceRecord> records;
	records.push_back(record_diffuse(0.45f, 0.45f, 0.45f));

	for (u32 j = 0u; j < gridZ; ++j) {
		for (u32 i = 0u; i < gridX; ++i) {
			f32 const x = (
				(static_cast<f32>(i) - static_cast<f32>(gridX - 1u) / 2.0f)
				* spacing
			);
			f32 const z = -2.0f + static_cast<f32>(j) * spacing;
			// traveling wave of bounce phases: a frozen frame of balls at
			// every point of the bounce arc
			f32 const phase = 0.55f * static_cast<f32>(i)
				+ 0.35f * static_cast<f32>(j);
			f32 const y = radius + bounceHeight * std::abs(std::sin(phase));

			u32 const customIndex = 1u + j * gridX + i;
			instances.push_back({
				1u,
				transform_scale_translate(radius, x, y, z),
				customIndex,
			});

			InstanceRecord rec = record_diffuse(0.0f, 0.0f, 0.0f);
			if ((i * 7u + j * 13u) % 11u == 0u) {
				// metallic accents scattered through the field
				rec.baseColor[0] = 0.95f;
				rec.baseColor[1] = 0.93f;
				rec.baseColor[2] = 0.88f;
				rec.baseMetalness = 1.0f;
				rec.specularWeight = 1.0f;
				rec.specularRoughness = 0.15f;
			} else {
				// cosine palette over the bounce phase: color encodes the
				// wave the heights follow
				f32 const tau = 6.28318530718f;
				rec.baseColor[0] = 0.55f + 0.40f * std::cos(phase);
				rec.baseColor[1] = 0.55f + 0.40f * std::cos(phase + tau / 3.0f);
				rec.baseColor[2] = 0.55f + 0.40f * std::cos(
					phase + 2.0f * tau / 3.0f
				);
				rec.specularWeight = 0.3f;
				rec.specularRoughness = 0.35f;
			}
			rec.flags = skFlagSphere;
			records.push_back(rec);
		}
	}

	auto const scene = scene_create(meshes, instances);
	vkof::acceleration_structure_set_tlas(scene.tlas);
	records[0].vertexVa = vkof::buffer_virtual_address(
		scene.vertexBuffers[0]
	);
	records[0].indexVa = vkof::buffer_virtual_address(
		scene.indexBuffers[0]
	);

	Camera const camera = camera_look_at(
		0.0f, 3.5f, -6.0f,
		0.0f, 0.5f, 6.0f,
		0.62f, 0.62f * 504.0f / 896.0f
	);

	auto const result = render_scene({
		.pipeline = pl,
		.records = &records,
		.camera = camera,
		.width = width,
		.height = height,
		.sampleCount = sampleCount,
		.bounceCount = bounceCount,
		.sky = true,
		.kullaContyHandle = energyTables.kullaContyEnergyHandle,
	});

	auto const stats = image_stats(result.rgb);
	CHECK(stats.nanCount == 0u);
	CHECK(stats.minValue >= 0.0f);
	CHECK(stats.mean > 0.05);
	CHECK(stats.mean < 5.0);

	// instance-index integrity across the whole tlas: every reported
	// primary hit is either the plane, a valid sphere index, or a miss
	u32 planeHits = 0u;
	u32 missCount = 0u;
	u32 invalidCount = 0u;
	std::vector<bool> seen(1u + sphereCount, false);
	for (u32 const inst : result.primaryHit) {
		if (inst == 0xFFFFFFFFu) {
			missCount++;
		} else if (inst > sphereCount) {
			invalidCount++;
		} else {
			seen[inst] = true;
			if (inst == 0u) { planeHits++; }
		}
	}
	u32 distinctSpheres = 0u;
	for (u32 i = 1u; i <= sphereCount; ++i) {
		if (seen[i]) { distinctSpheres++; }
	}
	CHECK(invalidCount == 0u);
	// the camera looks across the field to the horizon: the ground plane
	// fills the foreground, sky shows above the horizon, and well over a
	// thousand of the 3136 spheres are individually visible
	CHECK(planeHits > width * height / 10u);
	CHECK(missCount > width * height / 50u);
	CHECK(distinctSpheres > 1000u);

	save_png(
		result.rgb, width, height,
		DIELECTRIC_OUTPUT_DIR "dielectric-bouncing-spheres.png"
	);

	vkof::pipeline_destroy(pl);
	scene_destroy(scene);
	ponder::energy_tables_destroy(energyTables);
}

}
