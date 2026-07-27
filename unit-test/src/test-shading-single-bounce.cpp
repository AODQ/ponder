#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// layer 1 of the path-tracer verification effort: single-bounce direct
// lighting. builds directly on layer 0 (test-raycast-primitives.cpp) --
// real bvh geometry, one real cosine-weighted brdf-sampled bounce, no
// next-event-estimation (none exists in this harness yet) -- and checks
// the monte carlo mean against an independently-derived cpu quadrature of
// the parallel-plate radiative form factor, before any multi-bounce
// integration (layer 2) is trusted.
//
// scene: a large lambertian plane (y=0) under a small lambertian-emitting
// quad light (constant radiance Le on its front face, matching the
// wallEmission convention already used in test-cornell.cpp). exactly one
// bounce is traced past the primary hit, so the estimator's only nonzero
// contribution is "brdf-sampled ray lands on the light" -- isolating
// direct lighting from any indirect contribution by construction.
// ---------------------------------------------------------------------------

namespace {

constexpr f64 kPi = 3.14159265358979323846;

// -- cpu f64 quadrature: independent parallel-plate form-factor integral,
// -- not transcribed from the shader (which does monte carlo, not
// -- quadrature, so there is nothing to accidentally share code with)
//
// for two parallel horizontal planes (receiver normal (0,1,0), light
// normal (0,-1,0)) the two form-factor cosines are identical by symmetry
// (cosReceiver = cosLight = height / distance -- both normals are
// vertical, so both project the same way), collapsing the standard
// dE = Le * cosR * cosL * dA / r^2 differential to
// dE = Le * height^2 * dA / r^4, integrated by brute-force riemann sum
// over the light's rectangular extent (fine enough subdivision that
// truncation error is negligible next to monte carlo noise)
f64 quadrature_direct_lambertian(
	f64 const px, f64 const pz,
	f64 const lightCx, f64 const lightCz, f64 const lightHeight,
	f64 const lightHalfX, f64 const lightHalfZ,
	f64 const rho, f64 const le,
	u32 const subdiv
) {
	f64 const dx = 2.0 * lightHalfX / subdiv;
	f64 const dz = 2.0 * lightHalfZ / subdiv;
	f64 const dA = dx * dz;
	f64 sum = 0.0;
	for (u32 i = 0u; i < subdiv; ++i) {
		f64 const sx = (
			lightCx - lightHalfX + (static_cast<f64>(i) + 0.5) * dx
		);
		f64 const ddx = sx - px;
		for (u32 j = 0u; j < subdiv; ++j) {
			f64 const sz = (
				lightCz - lightHalfZ + (static_cast<f64>(j) + 0.5) * dz
			);
			f64 const ddz = sz - pz;
			f64 const r2 = (
				ddx * ddx + ddz * ddz + lightHeight * lightHeight
			);
			sum += (
				lightHeight * lightHeight * dA / (r2 * r2)
			);
		}
	}
	return (rho / kPi) * le * sum;
}

// -- gpu-side plumbing, mirrors single_bounce_direct_light.comp exactly

struct ShadePoint {
	f32 originX, originY, originZ;
	f32 dirX, dirY, dirZ;
};
static_assert(sizeof(ShadePoint) == 24u);

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

struct Vertex { f32 x, y, z; };

f32m44 const skIdentity = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
};

struct Scene {
	vkof::Buffer planeVb, planeIb, lightVb, lightIb;
	vkof::AccelerationStructureBlas planeBlas, lightBlas;
	vkof::AccelerationStructureTlas tlas;
};

Scene scene_create(
	f32 const lightCx, f32 const lightCz, f32 const lightHeight,
	f32 const lightHalfX, f32 const lightHalfZ
) {
	Scene scene {};

	Vertex const planeVerts[4] {
		{ -20.0f, 0.0f, -20.0f }, { 20.0f, 0.0f, -20.0f },
		{ 20.0f, 0.0f, 20.0f }, { -20.0f, 0.0f, 20.0f },
	};
	u32 const quadIndices[6] { 0u, 1u, 2u, 0u, 2u, 3u };
	scene.planeVb = upload_bytes(planeVerts, sizeof(planeVerts));
	scene.planeIb = upload_bytes(quadIndices, sizeof(quadIndices));
	scene.planeBlas = vkof::blas_create({
		.positionVa = vkof::buffer_virtual_address(scene.planeVb),
		.vertexCount = 4u,
		.indexVa = vkof::buffer_virtual_address(scene.planeIb),
		.triangleCount = 2u,
	});
	REQUIRE(scene.planeBlas.id != 0);

	Vertex const lightVerts[4] {
		{ lightCx - lightHalfX, lightHeight, lightCz - lightHalfZ },
		{ lightCx + lightHalfX, lightHeight, lightCz - lightHalfZ },
		{ lightCx + lightHalfX, lightHeight, lightCz + lightHalfZ },
		{ lightCx - lightHalfX, lightHeight, lightCz + lightHalfZ },
	};
	scene.lightVb = upload_bytes(lightVerts, sizeof(lightVerts));
	scene.lightIb = upload_bytes(quadIndices, sizeof(quadIndices));
	scene.lightBlas = vkof::blas_create({
		.positionVa = vkof::buffer_virtual_address(scene.lightVb),
		.vertexCount = 4u,
		.indexVa = vkof::buffer_virtual_address(scene.lightIb),
		.triangleCount = 2u,
	});
	REQUIRE(scene.lightBlas.id != 0);

	scene.tlas = vkof::tlas_create({ .maxInstances = 2u });
	REQUIRE(scene.tlas.id != 0);

	vkof::TlasInstance const instances[2] {
		{ scene.planeBlas, skIdentity, 0u, 0xFFu },
		{ scene.lightBlas, skIdentity, 1u, 0xFFu },
	};
	auto node = vkof::render_node_create(
		{ .queue = vkof::CommandQueue::compute }
	);
	vkof::render_node_callback({
		.node = node,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			vkof::tlas_build(
				cmd, scene.tlas,
				srat::slice<vkof::TlasInstance const>(instances, 2u)
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

void scene_destroy(Scene const & scene) {
	vkof::tlas_destroy(scene.tlas);
	vkof::blas_destroy(scene.lightBlas);
	vkof::blas_destroy(scene.planeBlas);
	vkof::buffer_destroy(scene.lightIb);
	vkof::buffer_destroy(scene.lightVb);
	vkof::buffer_destroy(scene.planeIb);
	vkof::buffer_destroy(scene.planeVb);
}

}

TEST_SUITE("[headless]") {

TEST_CASE("shading: single-bounce direct light matches cpu quadrature") {
	f32 const lightHeight = 2.0f;
	f32 const lightHalfX = 0.5f;
	f32 const lightHalfZ = 0.5f;
	f32 const rho = 0.9f;
	f32 const le = 5.0f;
	u32 const sampleCount = 2000000u;

	auto const scene = scene_create(
		0.0f, 0.0f, lightHeight, lightHalfX, lightHalfZ
	);
	vkof::acceleration_structure_set_tlas(scene.tlas);

	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "single_bounce_direct_light.comp",
	});
	REQUIRE(pl.id != 0);

	// shading points at varying horizontal offset from directly under the
	// light center, so both near-normal and oblique incidence are covered
	struct Point { f32 x, z; };
	std::vector<Point> const points {
		{ 0.0f, 0.0f }, { 0.3f, 0.2f }, { 0.8f, -0.5f },
		{ 1.5f, 0.0f }, { -0.6f, 1.0f }, { 2.5f, 2.5f },
	};

	std::vector<ShadePoint> probes;
	for (auto const & pt : points) {
		// primary ray origin must stay below the light's height (2.0):
		// several shading points sit directly under the light's
		// footprint, and a camera above the light would hit the light
		// quad itself first, not the plane
		probes.push_back(ShadePoint {
			pt.x, 1.0f, pt.z,
			0.0f, -1.0f, 0.0f,
		});
	}
	auto shadeBuf = upload_bytes(
		probes.data(), probes.size() * sizeof(ShadePoint)
	);
	auto outBuf = vkof::buffer_create({
		.byteCount = probes.size() * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push {
		u64 shadeVa;
		u64 outVa;
		u32 shadeCount;
		u32 sampleCount;
		f32 albedo;
		f32 emission;
	};
	Push const push {
		.shadeVa = vkof::buffer_virtual_address(shadeBuf),
		.outVa = vkof::buffer_virtual_address(outBuf),
		.shadeCount = static_cast<u32>(probes.size()),
		.sampleCount = sampleCount,
		.albedo = rho,
		.emission = le,
	};
	test::dispatch(
		pl, push, (static_cast<u32>(probes.size()) + 63u) / 64u
	);
	test::gpu_wait();

	auto const result = test::readback<f32>(
		outBuf, 0u, static_cast<u32>(probes.size())
	);

	for (u32 i = 0u; i < points.size(); ++i) {
		CAPTURE(points[i].x);
		CAPTURE(points[i].z);
		f64 const expected = quadrature_direct_lambertian(
			points[i].x, points[i].z,
			0.0, 0.0, lightHeight, lightHalfX, lightHalfZ,
			rho, le, /*subdiv=*/400u
		);
		// each sample is bernoulli: contributes exactly rho*le with
		// probability p = expected / (rho*le), else 0. use that analytic
		// p to bound the monte carlo standard error rather than an
		// arbitrary tolerance, then require agreement within 6 sigma
		// (generous margin against a flaky false failure) plus a small
		// absolute floor for the near-zero-probability oblique points
		f64 const p = expected / (static_cast<f64>(rho) * le);
		f64 const stderr_ = (
			static_cast<f64>(rho) * le
			* std::sqrt(std::max(p * (1.0 - p), 0.0) / sampleCount)
		);
		f64 const tolerance = std::max(6.0 * stderr_, 1e-5);
		f64 const actual = static_cast<f64>(result[i]);
		MESSAGE(
			"expected=", expected, " actual=", actual,
			" p=", p, " tolerance=", tolerance
		);
		CHECK(std::fabs(actual - expected) < tolerance);
	}

	vkof::pipeline_destroy(pl);
	vkof::buffer_destroy(outBuf);
	vkof::buffer_destroy(shadeBuf);
	scene_destroy(scene);
}

} // TEST_SUITE("[headless]")
