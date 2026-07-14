#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// layer 0 of the path-tracer verification effort: the raw ray-casting
// primitives everything else is built on -- ray-triangle hit distance,
// barycentric hit position, tlas instance transforms, instance/primitive
// indexing, and self-intersection epsilon handling (shadow acne vs corner
// tunneling). intersection itself runs on fixed-function hardware bvh
// traversal (VK_KHR_ray_query), so there is no gpu/cpu shared code path to
// accidentally crosscheck against -- the cpu-side moeller-trumbore
// reference below is derived from scratch and IS the independent oracle
// every test in this file compares against, filling the role the
// materialx crosscheck plays in the openpbr port-verification series.
//
// single generic probe shader (raycast_probe.comp) drives every test: one
// ray in, full hit state out (hit/t/instance/primitive/bary/objectToWorld).
// ---------------------------------------------------------------------------

namespace {

// -- cpu f64 reference: independent moeller-trumbore, not transcribed from
// -- any gpu code (hardware bvh traversal has no such code to transcribe)

struct RefVec3 { f64 x, y, z; };

RefVec3 ref_sub(RefVec3 const a, RefVec3 const b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}
RefVec3 ref_add(RefVec3 const a, RefVec3 const b) {
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}
RefVec3 ref_scale(RefVec3 const a, f64 const s) {
	return { a.x * s, a.y * s, a.z * s };
}
RefVec3 ref_cross(RefVec3 const a, RefVec3 const b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x,
	};
}
f64 ref_dot(RefVec3 const a, RefVec3 const b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
f64 ref_length(RefVec3 const a) {
	return std::sqrt(ref_dot(a, a));
}
RefVec3 ref_normalize(RefVec3 const a) {
	f64 const l = ref_length(a);
	return { a.x / l, a.y / l, a.z / l };
}

struct RefHit {
	bool hit;
	f64 t, u, v;
};

// moeller-trumbore ray-triangle intersection. u, v are the same
// barycentric convention vulkan reports: hitPos = (1-u-v)*p0 + u*p1 + v*p2
RefHit ref_ray_triangle(
	RefVec3 const origin, RefVec3 const dir,
	RefVec3 const p0, RefVec3 const p1, RefVec3 const p2,
	f64 const tMin, f64 const tMax
) {
	RefVec3 const edge1 = ref_sub(p1, p0);
	RefVec3 const edge2 = ref_sub(p2, p0);
	RefVec3 const pvec = ref_cross(dir, edge2);
	f64 const det = ref_dot(edge1, pvec);
	if (std::fabs(det) < 1e-12) { return { false, 0.0, 0.0, 0.0 }; }
	f64 const invDet = 1.0 / det;
	RefVec3 const tvec = ref_sub(origin, p0);
	f64 const u = ref_dot(tvec, pvec) * invDet;
	if (u < 0.0 || u > 1.0) { return { false, 0.0, 0.0, 0.0 }; }
	RefVec3 const qvec = ref_cross(tvec, edge1);
	f64 const v = ref_dot(dir, qvec) * invDet;
	if (v < 0.0 || u + v > 1.0) { return { false, 0.0, 0.0, 0.0 }; }
	f64 const t = ref_dot(edge2, qvec) * invDet;
	if (t < tMin || t > tMax) { return { false, 0.0, 0.0, 0.0 }; }
	return { true, t, u, v };
}

struct RefTriangle {
	RefVec3 p0, p1, p2;
	u32 instanceCustomIndex;
	u32 primitiveIndex;
};

struct RefSceneHit {
	bool hit;
	f64 t;
	u32 instanceCustomIndex;
	u32 primitiveIndex;
};

// closest-hit across every candidate triangle, matching the
// gl_RayFlagsOpaqueEXT (no early-out) semantics the probe shader uses
RefSceneHit ref_scene_intersect(
	std::vector<RefTriangle> const & tris,
	RefVec3 const origin, RefVec3 const dir,
	f64 const tMin, f64 const tMax
) {
	RefSceneHit best { false, tMax, 0u, 0u };
	for (auto const & tri : tris) {
		RefHit const h = ref_ray_triangle(
			origin, dir, tri.p0, tri.p1, tri.p2, tMin, best.t
		);
		if (h.hit) {
			best = {
				true, h.t, tri.instanceCustomIndex, tri.primitiveIndex
			};
		}
	}
	return best;
}

// -- gpu-side probe plumbing, mirrors raycast_probe.comp exactly

struct RayProbe {
	f32 originX, originY, originZ;
	f32 tMin;
	f32 dirX, dirY, dirZ;
	f32 tMax;
	u32 cullMask;
};
static_assert(sizeof(RayProbe) == 36u);

struct RayHit {
	u32 hit;
	f32 t;
	u32 instanceCustomIndex;
	u32 primitiveIndex;
	f32 baryX;
	f32 baryY;
	f32 objectToWorld[12];
};
static_assert(sizeof(RayHit) == 72u);

RayProbe make_probe(
	RefVec3 const origin, RefVec3 const dir,
	f32 const tMin, f32 const tMax, u32 const cullMask = 0xFFu
) {
	return RayProbe {
		static_cast<f32>(origin.x),
		static_cast<f32>(origin.y),
		static_cast<f32>(origin.z),
		tMin,
		static_cast<f32>(dir.x),
		static_cast<f32>(dir.y),
		static_cast<f32>(dir.z),
		tMax,
		cullMask,
	};
}

vkof::Buffer upload_bytes(void const * data, u64 const byteCount) {
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

std::vector<RayHit> run_probes(
	vkof::AccelerationStructureTlas const tlas,
	std::vector<RayProbe> const & probes
) {
	vkof::acceleration_structure_set_tlas(tlas);

	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "raycast_probe.comp",
	});
	REQUIRE(pl.id != 0);

	auto rayBuf = upload_bytes(
		probes.data(), probes.size() * sizeof(RayProbe)
	);
	auto outBuf = vkof::buffer_create({
		.byteCount = probes.size() * sizeof(RayHit),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push {
		u64 rayVa;
		u64 outVa;
		u32 rayCount;
	};
	Push const push {
		.rayVa = vkof::buffer_virtual_address(rayBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.rayCount = static_cast<u32>(probes.size()),
	};
	test::dispatch(
		pl, push, (static_cast<u32>(probes.size()) + 63u) / 64u
	);
	test::gpu_wait();

	auto const result = test::readback<RayHit>(
		outBuf, 0u, static_cast<u32>(probes.size())
	);

	vkof::pipeline_destroy(pl);
	vkof::buffer_destroy(outBuf);
	vkof::buffer_destroy(rayBuf);
	return result;
}

// -- scene builder: N meshes, each instanced with its own transform

struct Vertex { f32 x, y, z; };

struct MeshDef {
	std::vector<Vertex> vertices;
	std::vector<u32> indices;
};

struct InstanceDef {
	u32 meshIndex;
	f32m44 transform;
	u32 customIndex;
	u32 rayMask { 0xFFu };
};

struct RaycastScene {
	std::vector<vkof::Buffer> vertexBuffers;
	std::vector<vkof::Buffer> indexBuffers;
	std::vector<vkof::AccelerationStructureBlas> blases;
	vkof::AccelerationStructureTlas tlas;
};

f32m44 const skIdentity = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
};

RaycastScene scene_create(
	std::vector<MeshDef> const & meshes,
	std::vector<InstanceDef> const & instances
) {
	RaycastScene scene {};
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
	test::gpu_wait();

	return scene;
}

void scene_destroy(RaycastScene const & scene) {
	vkof::tlas_destroy(scene.tlas);
	for (auto const & blas : scene.blases) { vkof::blas_destroy(blas); }
	for (auto const & buf : scene.indexBuffers) { vkof::buffer_destroy(buf); }
	for (auto const & buf : scene.vertexBuffers) { vkof::buffer_destroy(buf); }
}

// column-major yaw+pitch rotation, then translation (f32m44 is
// m[col*4+row], matching test-cornell.cpp's box_transform convention)
f32m44 transform_rotate_translate(
	f32 const yaw, f32 const pitch, RefVec3 const translate
) {
	f32 const cy = std::cos(yaw), sy = std::sin(yaw);
	f32 const cp = std::cos(pitch), sp = std::sin(pitch);
	// Ry(yaw) * Rx(pitch)
	f32 const m00 = cy, m01 = sy * sp, m02 = sy * cp;
	f32 const m10 = 0.0f, m11 = cp, m12 = -sp;
	f32 const m20 = -sy, m21 = cy * sp, m22 = cy * cp;
	return f32m44 {
		m00, m10, m20, 0.0f,
		m01, m11, m21, 0.0f,
		m02, m12, m22, 0.0f,
		static_cast<f32>(translate.x),
		static_cast<f32>(translate.y),
		static_cast<f32>(translate.z),
		1.0f,
	};
}

f32m44 transform_scale_rotate_translate(
	RefVec3 const scale, f32 const yaw, f32 const pitch,
	RefVec3 const translate
) {
	f32m44 rt = transform_rotate_translate(yaw, pitch, translate);
	// scale the rotation columns (pre-scale in object space)
	for (u32 col = 0u; col < 3u; ++col) {
		f32 const s = (
			col == 0u
			? static_cast<f32>(scale.x)
			: col == 1u
			? static_cast<f32>(scale.y)
			: static_cast<f32>(scale.z)
		);
		rt.m[col * 4u + 0u] *= s;
		rt.m[col * 4u + 1u] *= s;
		rt.m[col * 4u + 2u] *= s;
	}
	return rt;
}

RefVec3 transform_point(f32m44 const & m, RefVec3 const p) {
	return {
		m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z + m.m[12],
		m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z + m.m[13],
		m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14],
	};
}

}

TEST_SUITE("[headless]") {

// ---------------------------------------------------------------------------
// hit distance: gpu-reported t against an independently-derived cpu
// moeller-trumbore reference, over a battery of angles/origins including
// grazing and miss cases
// ---------------------------------------------------------------------------

TEST_CASE("raycast: hit t matches closed-form moeller-trumbore") {
	RefVec3 const p0 { -0.5, -0.5, 1.0 };
	RefVec3 const p1 { 0.5, -0.5, 1.0 };
	RefVec3 const p2 { 0.0, 0.5, 1.0 };

	std::vector<MeshDef> const meshes {
		{
			{ { -0.5f, -0.5f, 1.0f }, { 0.5f, -0.5f, 1.0f },
			{ 0.0f, 0.5f, 1.0f } },
			{ 0u, 1u, 2u },
		},
	};
	std::vector<InstanceDef> const instances {
		{ 0u, skIdentity, 0u },
	};
	auto const scene = scene_create(meshes, instances);

	struct Case { RefVec3 origin, dir; };
	std::vector<Case> cases;
	// dense grid of origins over and around the triangle, straight-on
	for (i32 ix = -6; ix <= 6; ++ix) {
		for (i32 iy = -6; iy <= 6; ++iy) {
			f32 const x = static_cast<f32>(ix) * 0.15f;
			f32 const y = static_cast<f32>(iy) * 0.15f;
			cases.push_back({ { x, y, 0.0 }, { 0.0, 0.0, 1.0 } });
		}
	}
	// oblique rays guaranteed to hit near the centroid
	RefVec3 const centroid {
		(p0.x + p1.x + p2.x) / 3.0, (p0.y + p1.y + p2.y) / 3.0,
		(p0.z + p1.z + p2.z) / 3.0,
	};
	for (i32 a = -8; a <= 8; ++a) {
		f32 const angle = static_cast<f32>(a) * 0.1f;
		RefVec3 const origin { std::sin(angle) * 2.0, std::cos(angle) * 2.0, -2.0 };
		RefVec3 const dir = ref_normalize(ref_sub(centroid, origin));
		cases.push_back({ origin, dir });
	}

	std::vector<RayProbe> probes;
	for (auto const & c : cases) {
		probes.push_back(make_probe(c.origin, c.dir, 0.0f, 100.0f));
	}
	auto const hits = run_probes(scene.tlas, probes);

	u32 hitCount = 0u, missCount = 0u;
	f64 maxTErr = 0.0;
	for (u32 i = 0u; i < cases.size(); ++i) {
		RefHit const ref = ref_ray_triangle(
			cases[i].origin, cases[i].dir, p0, p1, p2, 0.0, 100.0
		);
		CAPTURE(i);
		CHECK(hits[i].hit == (ref.hit ? 1u : 0u));
		if (ref.hit && hits[i].hit) {
			hitCount++;
			maxTErr = std::max(
				maxTErr,
				std::fabs(static_cast<f64>(hits[i].t) - ref.t)
			);
		} else if (!ref.hit) {
			missCount++;
		}
	}
	CHECK(hitCount > 0u);
	CHECK(missCount > 0u);
	// f32 hardware traversal vs f64 cpu reference: expect sub-micro error
	CHECK(maxTErr < 1e-4);

	scene_destroy(scene);
}

// ---------------------------------------------------------------------------
// barycentric-interpolated hit position: reconstruct the hit point from
// gpu bary + known object-space triangle, and check it lands on the ray
// ---------------------------------------------------------------------------

TEST_CASE("raycast: barycentric hit position matches ray equation") {
	RefVec3 const p0 { -1.0, -1.0, 2.0 };
	RefVec3 const p1 { 2.0, -0.5, 2.3 };
	RefVec3 const p2 { -0.3, 1.5, 1.7 };

	std::vector<MeshDef> const meshes {
		{
			{
				{ -1.0f, -1.0f, 2.0f },
				{ 2.0f, -0.5f, 2.3f },
				{ -0.3f, 1.5f, 1.7f },
			},
			{ 0u, 1u, 2u },
		},
	};
	std::vector<InstanceDef> const instances { { 0u, skIdentity, 0u } };
	auto const scene = scene_create(meshes, instances);

	RefVec3 const centroid {
		(p0.x + p1.x + p2.x) / 3.0, (p0.y + p1.y + p2.y) / 3.0,
		(p0.z + p1.z + p2.z) / 3.0,
	};
	std::vector<RayProbe> probes;
	std::vector<RefVec3> origins, dirs;
	for (i32 a = 0; a < 24; ++a) {
		f32 const angle = static_cast<f32>(a) * 0.26f;
		f32 const elev = static_cast<f32>((a % 5)) * 0.2f - 0.4f;
		RefVec3 const origin {
			std::sin(angle) * 3.0, elev * 3.0 - 1.0, std::cos(angle) * 3.0 - 3.0
		};
		RefVec3 const dir = ref_normalize(ref_sub(centroid, origin));
		origins.push_back(origin);
		dirs.push_back(dir);
		probes.push_back(make_probe(origin, dir, 0.0f, 100.0f));
	}
	auto const hits = run_probes(scene.tlas, probes);

	u32 hitCount = 0u;
	f64 maxPosErr = 0.0;
	for (u32 i = 0u; i < probes.size(); ++i) {
		CAPTURE(i);
		if (hits[i].hit == 0u) { continue; }
		hitCount++;
		f64 const u = hits[i].baryX;
		f64 const v = hits[i].baryY;
		RefVec3 const baryPos {
			(1.0 - u - v) * p0.x + u * p1.x + v * p2.x,
			(1.0 - u - v) * p0.y + u * p1.y + v * p2.y,
			(1.0 - u - v) * p0.z + u * p1.z + v * p2.z,
		};
		RefVec3 const rayPos = ref_add(
			origins[i], ref_scale(dirs[i], hits[i].t)
		);
		f64 const err = ref_length(ref_sub(baryPos, rayPos));
		maxPosErr = std::max(maxPosErr, err);
	}
	CHECK(hitCount > 0u);
	CHECK(maxPosErr < 1e-4);

	scene_destroy(scene);
}

// ---------------------------------------------------------------------------
// tlas instance transforms: translation-only, rotation-only, and combined
// scale+rotate+translate, cross-checked against a cpu-transformed triangle
// and against the objectToWorld matrix rayquery reports back
// ---------------------------------------------------------------------------

TEST_CASE("raycast: instance transforms reproduce cpu-transformed hits") {
	RefVec3 const objP0 { -0.5, -0.5, 0.0 };
	RefVec3 const objP1 { 0.5, -0.5, 0.0 };
	RefVec3 const objP2 { 0.0, 0.5, 0.0 };

	std::vector<MeshDef> const meshes {
		{
			{
				{ -0.5f, -0.5f, 0.0f },
				{ 0.5f, -0.5f, 0.0f },
				{ 0.0f, 0.5f, 0.0f },
			},
			{ 0u, 1u, 2u },
		},
	};

	struct Config { char const * name; f32m44 transform; };
	std::vector<Config> const configs {
		{ "translate-only", { 1,0,0,0, 0,1,0,0, 0,0,1,0, 3,-2,5,1 } },
		{
			"rotate-only",
			transform_rotate_translate(0.7f, 0.3f, { 0.0, 0.0, 0.0 }),
		},
		{
			"scale-rotate-translate",
			transform_scale_rotate_translate(
				{ 2.0, 0.5, 1.3 }, 1.1f, -0.4f, { -1.0, 4.0, 2.0 }
			),
		},
	};

	std::vector<InstanceDef> instances;
	for (u32 i = 0u; i < configs.size(); ++i) {
		instances.push_back({ 0u, configs[i].transform, i, 0xFFu });
	}
	auto const scene = scene_create(meshes, instances);

	// fire one ray per instance, straight down -z at the world-space
	// centroid of that instance's transformed triangle
	std::vector<RayProbe> probes;
	for (u32 i = 0u; i < configs.size(); ++i) {
		RefVec3 const wp0 = transform_point(configs[i].transform, objP0);
		RefVec3 const wp1 = transform_point(configs[i].transform, objP1);
		RefVec3 const wp2 = transform_point(configs[i].transform, objP2);
		RefVec3 const centroid {
			(wp0.x + wp1.x + wp2.x) / 3.0,
			(wp0.y + wp1.y + wp2.y) / 3.0,
			(wp0.z + wp1.z + wp2.z) / 3.0,
		};
		RefVec3 const origin = ref_add(centroid, RefVec3 { 0.0, 0.0, 10.0 });
		RefVec3 const dir { 0.0, 0.0, -1.0 };
		probes.push_back(make_probe(origin, dir, 0.0f, 100.0f));

		RefHit const ref = ref_ray_triangle(
			origin, dir, wp0, wp1, wp2, 0.0, 100.0
		);
		REQUIRE(ref.hit);
	}
	auto const hits = run_probes(scene.tlas, probes);

	for (u32 i = 0u; i < configs.size(); ++i) {
		CAPTURE(configs[i].name);
		REQUIRE(hits[i].hit == 1u);
		CHECK(hits[i].instanceCustomIndex == i);

		RefVec3 const wp0 = transform_point(configs[i].transform, objP0);
		RefVec3 const wp1 = transform_point(configs[i].transform, objP1);
		RefVec3 const wp2 = transform_point(configs[i].transform, objP2);
		RefVec3 const origin {
			probes[i].originX, probes[i].originY, probes[i].originZ
		};
		RefVec3 const dir { probes[i].dirX, probes[i].dirY, probes[i].dirZ };
		RefHit const ref = ref_ray_triangle(
			origin, dir, wp0, wp1, wp2, 0.0, 100.0
		);
		REQUIRE(ref.hit);
		CHECK(std::fabs(static_cast<f64>(hits[i].t) - ref.t) < 1e-3);

		// objectToWorld rayquery reports must equal the supplied instance
		// transform's affine part exactly (mat4x3, column-major)
		f32m44 const & m = configs[i].transform;
		f64 maxMatErr = 0.0;
		for (u32 col = 0u; col < 4u; ++col) {
			for (u32 row = 0u; row < 3u; ++row) {
				f32 const expected = m.m[col * 4u + row];
				f32 const actual = hits[i].objectToWorld[col * 3u + row];
				maxMatErr = std::max(
					maxMatErr,
					std::fabs(static_cast<f64>(actual - expected))
				);
			}
		}
		CHECK(maxMatErr < 1e-4);
	}

	scene_destroy(scene);
}

// ---------------------------------------------------------------------------
// multi-instance / multi-primitive indexing: several instances, each a
// multi-triangle mesh, ray targeted at a specific (instance, primitive)
// ---------------------------------------------------------------------------

TEST_CASE("raycast: instanceCustomIndex and primitiveIndex are exact") {
	// a 2x2 grid of quads (4 tris via 2 tris per quad... use a simple
	// strip of 4 independent triangles spread along x, one blas)
	std::vector<Vertex> verts;
	std::vector<u32> indices;
	for (u32 t = 0u; t < 4u; ++t) {
		f32 const cx = static_cast<f32>(t) * 2.0f;
		verts.push_back({ cx - 0.5f, -0.5f, 0.0f });
		verts.push_back({ cx + 0.5f, -0.5f, 0.0f });
		verts.push_back({ cx, 0.5f, 0.0f });
		indices.push_back(t * 3u + 0u);
		indices.push_back(t * 3u + 1u);
		indices.push_back(t * 3u + 2u);
	}
	std::vector<MeshDef> const meshes { { verts, indices } };

	// three instances of the same 4-triangle mesh, offset along y and
	// tagged with distinct, non-sequential custom indices
	std::vector<InstanceDef> const instances {
		{ 0u, { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 }, 42u },
		{ 0u, { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,10,0,1 }, 7u },
		{ 0u, { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,20,0,1 }, 99u },
	};
	auto const scene = scene_create(meshes, instances);

	std::vector<RayProbe> probes;
	struct Expect { u32 customIndex, primitiveIndex; };
	std::vector<Expect> expected;
	f32 const instanceY[3] = { 0.0f, 10.0f, 20.0f };
	u32 const instanceCustom[3] = { 42u, 7u, 99u };
	for (u32 inst = 0u; inst < 3u; ++inst) {
		for (u32 tri = 0u; tri < 4u; ++tri) {
			f32 const cx = static_cast<f32>(tri) * 2.0f;
			RefVec3 const origin {
				cx, static_cast<f64>(instanceY[inst]) + 0.0, 5.0
			};
			RefVec3 const dir { 0.0, 0.0, -1.0 };
			probes.push_back(make_probe(origin, dir, 0.0f, 100.0f));
			expected.push_back({ instanceCustom[inst], tri });
		}
	}
	auto const hits = run_probes(scene.tlas, probes);

	for (u32 i = 0u; i < probes.size(); ++i) {
		CAPTURE(i);
		REQUIRE(hits[i].hit == 1u);
		CHECK(hits[i].instanceCustomIndex == expected[i].customIndex);
		CHECK(hits[i].primitiveIndex == expected[i].primitiveIndex);
	}

	scene_destroy(scene);
}

// ---------------------------------------------------------------------------
// self-intersection epsilon handling, failure mode 1: corner tunneling.
// a shading point sits at origin offset epsOffset along the floor's
// normal, at x = +delta from a perpendicular wall occupying the x=0
// plane. a ray parallel to the floor (direction -x) must cross the wall
// at true distance ~delta; if tMin exceeds that, the wall is skipped and
// the ray tunnels through -- reproduces the cornell-box corner-leak bug
// in isolation, as a swept regression rather than tribal knowledge.
// ---------------------------------------------------------------------------

TEST_CASE("raycast: corner-tunneling epsilon/tMin sweep") {
	// floor: y=0 plane, x in [-2,2], z in [-2,2]
	// wall: x=0 plane, y in [0,2], z in [-2,2] (perpendicular to floor)
	std::vector<MeshDef> const meshes {
		{
			{ { -2,0,-2 }, { 2,0,-2 }, { 2,0,2 }, { -2,0,2 } },
			{ 0,1,2, 0,2,3 },
		},
		{
			{ { 0,0,-2 }, { 0,2,-2 }, { 0,2,2 }, { 0,0,2 } },
			{ 0,1,2, 0,2,3 },
		},
	};
	std::vector<InstanceDef> const instances {
		{ 0u, skIdentity, 0u },
		{ 1u, skIdentity, 1u },
	};
	auto const scene = scene_create(meshes, instances);

	u32 const deltaSteps = 24u, tMinSteps = 24u;
	// delta (distance from the corner, along the floor toward the wall)
	// swept from tiny to comfortably-clear-of-the-wall
	f32 const deltaMax = 0.01f;
	f32 const tMinMax = 0.01f;

	std::vector<RayProbe> probes;
	for (u32 di = 0u; di < deltaSteps; ++di) {
		f32 const delta = (
			(static_cast<f32>(di) + 0.5f) / deltaSteps * deltaMax
		);
		for (u32 ti = 0u; ti < tMinSteps; ++ti) {
			// offset from the delta grid's cell-center sampling (di+0.5)
			// so no (delta, tMinVal) pair ever lands on the exact float
			// value: right at tMinVal == delta the true crossing sits
			// exactly on the tMin boundary, and whether the hardware
			// intersector treats tMin as inclusive or exclusive there is
			// not something this test should assert either way
			f32 const tMinVal = (
				static_cast<f32>(ti) / tMinSteps * tMinMax
			);
			// origin sits epsOffset above the floor (prevents floor
			// self-intersection, irrelevant to the wall test) at x=delta
			f32 const epsOffset = 0.001f;
			probes.push_back(make_probe(
				RefVec3 { delta, epsOffset, 0.0 },
				RefVec3 { -1.0, 0.0, 0.0 },
				tMinVal, 100.0f
			));
		}
	}
	auto const hits = run_probes(scene.tlas, probes);

	std::vector<f32> heat(deltaSteps * tMinSteps);
	u32 idx = 0u;
	u32 tunneledAtSmallTMin = 0u;
	for (u32 di = 0u; di < deltaSteps; ++di) {
		for (u32 ti = 0u; ti < tMinSteps; ++ti, ++idx) {
			f32 const delta = (
				(static_cast<f32>(di) + 0.5f) / deltaSteps * deltaMax
			);
			// offset from the delta grid's cell-center sampling (di+0.5)
			// so no (delta, tMinVal) pair ever lands on the exact float
			// value: right at tMinVal == delta the true crossing sits
			// exactly on the tMin boundary, and whether the hardware
			// intersector treats tMin as inclusive or exclusive there is
			// not something this test should assert either way
			f32 const tMinVal = (
				static_cast<f32>(ti) / tMinSteps * tMinMax
			);
			bool const hitWall = (
				hits[idx].hit == 1u && hits[idx].instanceCustomIndex == 1u
			);
			heat[idx] = hitWall ? 1.0f : 0.0f;
			// analytic prediction: the wall is hit iff tMin <= delta
			// (true crossing distance along -x from x=delta to x=0)
			bool const expectHit = tMinVal <= delta;
			CAPTURE(delta);
			CAPTURE(tMinVal);
			CHECK(hitWall == expectHit);
			if (tMinVal > delta) { tunneledAtSmallTMin++; }
		}
	}
	// confirms the failure mode is real and reachable in this scene: at
	// least some (tMin > delta) configurations exist and do tunnel
	CHECK(tunneledAtSmallTMin > 0u);

	// the operating point actually used by cornell_render.comp
	// (tMin = 0.0) never tunnels regardless of delta
	{
		auto const zeroTMinProbe = make_probe(
			RefVec3 { deltaMax * 0.01, 0.001, 0.0 },
			RefVec3 { -1.0, 0.0, 0.0 }, 0.0f, 100.0f
		);
		auto const r = run_probes(scene.tlas, { zeroTMinProbe });
		CHECK(r[0].hit == 1u);
		CHECK(r[0].instanceCustomIndex == 1u);
	}

	CHECK(test::write_heatmap_png(
		heat, heat, heat, tMinSteps, deltaSteps,
		RAYCAST_OUTPUT_DIR "raycast-corner-tunneling.png"
	));

	scene_destroy(scene);
}

// ---------------------------------------------------------------------------
// self-intersection epsilon handling, failure mode 2: shadow acne. a
// shading point sits exactly on (or epsOffset above) a plane and fires a
// grazing ray nearly parallel to that same plane; the textbook failure is
// float round-off in the intersector re-hitting the same plane at t~0.
//
// result (2026-07-16): not reproduced. swept epsOffset x tMin over
// [0, 1e-3]^2 (576 configs) against a single large planar quad with a
// ~0.0002 rad grazing ray, including the fully degenerate case
// (epsOffset=0, tMin=0 exactly) -- every config reports a clean miss, 0
// spurious self-hits. this is a genuine negative result, not a weak test:
// this VK_KHR_ray_query implementation's hardware intersector evidently
// handles the coplanar/tMin=0 case robustly for a simple single-triangle
// plane at this scale, unlike naive software moeller-trumbore. left in as
// a standing regression -- if this ever starts failing (acneCount > 0),
// that is itself the notable event. not chased further with more
// adversarial geometry (denser meshes, steeper grazing, float-lossy
// world-space origins) per the user's call on 2026-07-16.
// ---------------------------------------------------------------------------

TEST_CASE("raycast: shadow-acne epsilon/tMin sweep") {
	// a single large plane, y=0, x/z in [-50,50] so grazing rays travel
	// far across it before any edge is reached
	std::vector<MeshDef> const meshes {
		{
			{ { -50,0,-50 }, { 50,0,-50 }, { 50,0,50 }, { -50,0,50 } },
			{ 0,1,2, 0,2,3 },
		},
	};
	std::vector<InstanceDef> const instances { { 0u, skIdentity, 0u } };
	auto const scene = scene_create(meshes, instances);

	u32 const epsSteps = 24u, tMinSteps = 24u;
	f32 const epsMax = 1e-3f;
	f32 const tMinMax = 1e-3f;
	// grazing angle: mostly along +x, tiny +y lift so it is a true miss
	// in exact arithmetic (never re-crosses y=0 within the plane extent)
	RefVec3 const dir = ref_normalize(RefVec3 { 1.0, 0.0002, 0.0 });

	std::vector<RayProbe> probes;
	for (u32 ei = 0u; ei < epsSteps; ++ei) {
		f32 const eps = static_cast<f32>(ei) / epsSteps * epsMax;
		for (u32 ti = 0u; ti < tMinSteps; ++ti) {
			f32 const tMinVal = static_cast<f32>(ti) / tMinSteps * tMinMax;
			probes.push_back(make_probe(
				RefVec3 { 0.0, eps, 0.0 }, dir, tMinVal, 40.0f
			));
		}
	}
	auto const hits = run_probes(scene.tlas, probes);

	std::vector<f32> heat(epsSteps * tMinSteps);
	u32 idx = 0u;
	u32 acneCount = 0u;
	u32 cleanCount = 0u;
	for (u32 ei = 0u; ei < epsSteps; ++ei) {
		for (u32 ti = 0u; ti < tMinSteps; ++ti, ++idx) {
			// acne: a hit reported at a suspiciously small t, well before
			// the ray's true re-crossing of y=0 (~ eps / 0.0002, which
			// for eps<=1e-3 is >=5 units away -- anything hit under 1.0
			// is spurious self-intersection, not the legitimate far
			// re-crossing)
			bool const spuriousHit = (
				hits[idx].hit == 1u && hits[idx].t < 1.0f
			);
			heat[idx] = spuriousHit ? 1.0f : 0.0f;
			if (spuriousHit) { acneCount++; } else { cleanCount++; }
		}
	}
	CHECK(cleanCount > 0u);
	// empirical baseline, not a guaranteed hardware invariant: this
	// VK_KHR_ray_query implementation has not been observed to produce
	// acne anywhere in this swept range (see header comment). if this
	// ever fails, that regression is itself the interesting finding
	CHECK(acneCount == 0u);
	MESSAGE(
		"acneCount=", acneCount, " cleanCount=", cleanCount,
		" (", epsSteps * tMinSteps, " total)"
	);
	{
		auto const worst = run_probes(
			scene.tlas,
			{ make_probe(RefVec3 { 0.0, 0.0, 0.0 }, dir, 0.0f, 40.0f) }
		);
		MESSAGE(
			"eps=0,tMin=0: hit=", worst[0].hit, " t=", worst[0].t
		);
	}

	// the operating point actually used by cornell_render.comp
	// (epsOffset = 0.001, tMin = 0.0) must be acne-free
	{
		auto const opProbe = make_probe(
			RefVec3 { 0.0, 0.001, 0.0 }, dir, 0.0f, 40.0f
		);
		auto const r = run_probes(scene.tlas, { opProbe });
		CHECK((r[0].hit == 0u || r[0].t > 1.0f));
	}

	CHECK(test::write_heatmap_png(
		heat, heat, heat, tMinSteps, epsSteps,
		RAYCAST_OUTPUT_DIR "raycast-shadow-acne.png"
	));

	scene_destroy(scene);
}

// ---------------------------------------------------------------------------
// watertightness: many rays crossing the shared internal edge of a
// two-triangle quad (single blas) must never miss
// ---------------------------------------------------------------------------

TEST_CASE("raycast: watertight across a shared internal edge") {
	std::vector<MeshDef> const meshes {
		{
			{ { -1,-1,0 }, { 1,-1,0 }, { 1,1,0 }, { -1,1,0 } },
			{ 0,1,2, 0,2,3 },
		},
	};
	std::vector<InstanceDef> const instances { { 0u, skIdentity, 0u } };
	auto const scene = scene_create(meshes, instances);

	// shared edge runs from (-1,-1,0) to (1,1,0); sample densely along its
	// interior (kept off the quad's own corners at t=+-1, where an offset
	// probe can legitimately land outside the quad entirely -- that is a
	// finite-extent miss, not a watertightness gap) and slightly to
	// either side, straight-on rays from +z
	std::vector<RayProbe> probes;
	u32 const steps = 200u;
	for (u32 i = 0u; i < steps; ++i) {
		f32 const t = (
			static_cast<f32>(i) / (steps - 1u) * 2.0f * 0.98f - 0.98f
		);
		for (f32 const off : { -1e-5f, 0.0f, 1e-5f }) {
			probes.push_back(make_probe(
				RefVec3 { static_cast<f64>(t + off), static_cast<f64>(t), 5.0 },
				RefVec3 { 0.0, 0.0, -1.0 }, 0.0f, 100.0f
			));
		}
	}
	auto const hits = run_probes(scene.tlas, probes);

	u32 missCount = 0u;
	for (auto const & h : hits) { if (h.hit == 0u) { missCount++; } }
	CHECK(missCount == 0u);

	scene_destroy(scene);
}

} // TEST_SUITE("[headless]")
