#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <ponder/energy-tables.hpp>
#include "util.hpp"
#include <vector>

// ---------------------------------------------------------------------------
// cornell box scene shared by the cornell test suite.
//
// interior box spanning [-1,1]^3 with the front face (z=-1) open; the camera
// sits outside at (0,0,-3) looking down +z. each wall is its own BLAS,
// instanced into the TLAS with instanceCustomIndex = wall index, so shaders
// can look up per-wall material data from the committed hit.
//
// wall quads extend to +-1.05 so adjacent walls overlap at the corners:
// watertightness is only guaranteed within a BLAS, not across BLASes, so
// without the overlap a ray hitting the exact seam between two walls could
// leak out of the box
// ---------------------------------------------------------------------------

namespace {

struct Vertex { f32 x, y, z; };

// wall indices, doubling as material indices in later tests
static u32 const skWallFloor = 0u;
static u32 const skWallCeiling = 1u;
static u32 const skWallBack = 2u;
static u32 const skWallLeft = 3u;
static u32 const skWallRight = 4u;
static u32 const skWallLight = 5u;
static u32 const skWallCount = 6u;
// the front wall (z=-1) closes the box for the furnace test only
static u32 const skWallFront = 6u;
static u32 const skWallCountClosed = 7u;

// overlapped wall extent, see header comment
static f32 const skE = 1.05f;

// each wall is a quad given by 4 corners, wound counter-clockwise as seen
// from inside the box so geometric normals face inward
static Vertex const skWallQuads[skWallCountClosed][4] = {
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
	// left wall (x=-1), red in later tests
	{
		{ -1.0f, -skE, -skE },
		{ -1.0f, -skE, skE },
		{ -1.0f, skE, skE },
		{ -1.0f, skE, -skE },
	},
	// right wall (x=+1), green in later tests
	{
		{ 1.0f, -skE, -skE },
		{ 1.0f, skE, -skE },
		{ 1.0f, skE, skE },
		{ 1.0f, -skE, skE },
	},
	// area light, slightly below the ceiling. wound so the geometric normal
	// points down (-y): the openpbr render emits from the front face only,
	// otherwise the 0.002 gap between light and ceiling forms a radiosity
	// cavity (L ~ rho * Le / (1 - rho_c * rho_l), ~25 for these albedos)
	// visible through the slit as pixels far brighter than the emitter
	{
		{ -0.25f, 0.998f, -0.25f },
		{ 0.25f, 0.998f, -0.25f },
		{ 0.25f, 0.998f, 0.25f },
		{ -0.25f, 0.998f, 0.25f },
	},
	// front wall (z=-1), furnace test only
	{
		{ -skE, -skE, -1.0f },
		{ -skE, skE, -1.0f },
		{ skE, skE, -1.0f },
		{ skE, -skE, -1.0f },
	},
};

static u32 const skQuadIndices[6] = { 0u, 1u, 2u, 0u, 2u, 3u };

// interior boxes (openpbr material test): instance indices continue after
// the walls. the front wall and the boxes are never used together
static u32 const skBoxTall = 6u;
static u32 const skBoxShort = 7u;
static u32 const skInstanceCountBoxes = 8u;
static u32 const skInstanceCountMax = 8u;

// unit cube corners; scaled per box at upload so the TLAS instance
// transform stays a pure rotation + translation and geometric normals
// transform exactly by the rotation part
static Vertex const skCubeVerts[8] = {
	{ -1.0f, -1.0f, -1.0f },
	{ 1.0f, -1.0f, -1.0f },
	{ 1.0f, 1.0f, -1.0f },
	{ -1.0f, 1.0f, -1.0f },
	{ -1.0f, -1.0f, 1.0f },
	{ 1.0f, -1.0f, 1.0f },
	{ 1.0f, 1.0f, 1.0f },
	{ -1.0f, 1.0f, 1.0f },
};
static u32 const skCubeIndices[36] = {
	0u, 2u, 1u, 0u, 3u, 2u,
	4u, 5u, 6u, 4u, 6u, 7u,
	0u, 1u, 5u, 0u, 5u, 4u,
	3u, 7u, 6u, 3u, 6u, 2u,
	0u, 4u, 7u, 0u, 7u, 3u,
	1u, 2u, 6u, 1u, 6u, 5u,
};

// half-extents of the two boxes; tall box reaches y=0.2, short y=-0.4
static Vertex const skBoxScale[2] = {
	{ 0.3f, 0.6f, 0.3f },
	{ 0.3f, 0.3f, 0.3f },
};
// yaw and floor placement, mimicking the classic cornell arrangement
static f32 const skBoxYaw[2] = { 0.30f, -0.35f };
static Vertex const skBoxTranslate[2] = {
	{ -0.38f, -0.4f, 0.38f },
	{ 0.42f, -0.7f, -0.30f },
};

// column-major yaw rotation + translation (f32m44 is m[col*4+row])
static f32m44 box_transform(u32 const box) {
	f32 const c = std::cos(skBoxYaw[box]);
	f32 const s = std::sin(skBoxYaw[box]);
	return f32m44 {
		c, 0.0f, -s, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		s, 0.0f, c, 0.0f,
		skBoxTranslate[box].x, skBoxTranslate[box].y, skBoxTranslate[box].z,
		1.0f,
	};
}

static f32m44 const skIdentity = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
};

// image dimensions for camera-based tests. deliberately non-square: with a
// square image the corner-pixel rays pass exactly through the box corner
// edges, where hits across two different BLASes have no watertightness
// guarantee
static u32 const skImageWidth = 64u;
static u32 const skImageHeight = 48u;

struct CornellScene {
	u32 instanceCount;
	vkof::Buffer vertexBuffers[skInstanceCountMax];
	vkof::Buffer indexBuffers[skInstanceCountMax];
	vkof::AccelerationStructureBlas blases[skInstanceCountMax];
	vkof::AccelerationStructureTlas tlas;
};

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

static CornellScene cornell_scene_create(
	bool const closedFront,
	bool const withBoxes = false
) {
	// the furnace's front wall and the interior boxes are never combined
	REQUIRE(!(closedFront && withBoxes));
	CornellScene scene {};
	u32 const wallCount = closedFront ? skWallCountClosed : skWallCount;
	scene.instanceCount = withBoxes ? skInstanceCountBoxes : wallCount;

	for (u32 wall = 0u; wall < wallCount; ++wall) {
		scene.vertexBuffers[wall] = upload_bytes(
			skWallQuads[wall], sizeof(skWallQuads[wall])
		);
		scene.indexBuffers[wall] = upload_bytes(
			skQuadIndices, sizeof(skQuadIndices)
		);
		scene.blases[wall] = vkof::blas_create({
			.positionVa = vkof::buffer_virtual_address(
				scene.vertexBuffers[wall]
			),
			.vertexCount = 4u,
			.indexVa = vkof::buffer_virtual_address(scene.indexBuffers[wall]),
			.triangleCount = 2u,
		});
		REQUIRE(scene.blases[wall].id != 0);
	}

	if (withBoxes) {
		for (u32 box = 0u; box < 2u; ++box) {
			Vertex scaled[8];
			for (u32 v = 0u; v < 8u; ++v) {
				scaled[v] = Vertex {
					skCubeVerts[v].x * skBoxScale[box].x,
					skCubeVerts[v].y * skBoxScale[box].y,
					skCubeVerts[v].z * skBoxScale[box].z,
				};
			}
			u32 const inst = skBoxTall + box;
			scene.vertexBuffers[inst] = upload_bytes(scaled, sizeof(scaled));
			scene.indexBuffers[inst] = upload_bytes(
				skCubeIndices, sizeof(skCubeIndices)
			);
			scene.blases[inst] = vkof::blas_create({
				.positionVa = vkof::buffer_virtual_address(
					scene.vertexBuffers[inst]
				),
				.vertexCount = 8u,
				.indexVa = vkof::buffer_virtual_address(
					scene.indexBuffers[inst]
				),
				.triangleCount = 12u,
			});
			REQUIRE(scene.blases[inst].id != 0);
		}
	}

	scene.tlas = vkof::tlas_create({ .maxInstances = scene.instanceCount });
	REQUIRE(scene.tlas.id != 0);

	vkof::TlasInstance instances[skInstanceCountMax];
	for (u32 inst = 0u; inst < scene.instanceCount; ++inst) {
		bool const isBox = withBoxes && inst >= skBoxTall;
		instances[inst] = vkof::TlasInstance {
			.blas = scene.blases[inst],
			.transform = isBox ? box_transform(inst - skBoxTall) : skIdentity,
			.instanceCustomIndex = inst,
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
					instances, scene.instanceCount
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

static void cornell_scene_destroy(CornellScene const & scene) {
	vkof::tlas_destroy(scene.tlas);
	for (u32 inst = 0u; inst < scene.instanceCount; ++inst) {
		vkof::blas_destroy(scene.blases[inst]);
		vkof::buffer_destroy(scene.indexBuffers[inst]);
		vkof::buffer_destroy(scene.vertexBuffers[inst]);
	}
}

}

TEST_SUITE("[headless]") {

TEST_CASE("cornell: primary ray visibility") {
	auto const scene = cornell_scene_create(/*closedFront=*/false);
	vkof::acceleration_structure_set_tlas(scene.tlas);

	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "cornell_visibility.comp",
	});
	REQUIRE(pl.id != 0);

	u32 const pixelCount = skImageWidth * skImageHeight;
	auto outBuf = test::make_buffer_u32(pixelCount);

	struct Push {
		u64 outVa;
		u32 width;
		u32 height;
	};
	Push const push {
		.outVa = vkof::buffer_virtual_address(outBuf),
		.width = skImageWidth,
		.height = skImageHeight,
	};
	test::dispatch(
		pl,
		push,
		skImageWidth / 8u,
		skImageHeight / 8u
	);
	test::gpu_wait();

	auto const result = test::readback<u32>(outBuf, 0u, pixelCount);

	// every pixel must hit a wall: the camera fov keeps all rays inside the
	// open front face, and the box is closed everywhere else along +z
	u32 wallHits[skWallCount] = {};
	u32 missCount = 0u;
	u32 invalidCount = 0u;
	for (u32 i = 0u; i < pixelCount; ++i) {
		if (result[i] == 0xFFFFFFFFu) {
			missCount++;
		} else if (result[i] >= skWallCount) {
			invalidCount++;
		} else {
			wallHits[result[i]]++;
		}
	}
	CHECK(missCount == 0u);
	CHECK(invalidCount == 0u);

	// every wall (including the light) is visible from the camera
	for (u32 wall = 0u; wall < skWallCount; ++wall) {
		CHECK(wallHits[wall] > 0u);
	}

	auto pixel = [&](u32 const px, u32 const py) {
		return result[py * skImageWidth + px];
	};
	// deterministic per-pixel checks; positions hand-verified against the
	// camera geometry (origin z=-3, tanHalfFov 0.4)
	CHECK(pixel(32u, 24u) == skWallBack);
	CHECK(pixel(0u, 24u) == skWallLeft);
	CHECK(pixel(63u, 24u) == skWallRight);
	CHECK(pixel(32u, 0u) == skWallCeiling);
	CHECK(pixel(32u, 47u) == skWallFloor);
	// this ray crosses the y=0.998 plane at z~=0.07, inside the light quad
	CHECK(pixel(32u, 4u) == skWallLight);

	vkof::pipeline_destroy(pl);
	vkof::buffer_destroy(outBuf);
	cornell_scene_destroy(scene);
}

TEST_CASE("cornell: closed box furnace") {
	auto const scene = cornell_scene_create(/*closedFront=*/true);
	vkof::acceleration_structure_set_tlas(scene.tlas);

	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "cornell_furnace.comp",
	});
	REQUIRE(pl.id != 0);

	u32 const pixelCount = skImageWidth * skImageHeight;
	auto outBuf = vkof::buffer_create({
		.byteCount = static_cast<u64>(pixelCount) * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push {
		u64 outVa;
		u32 width;
		u32 height;
	};
	Push const push {
		.outVa = vkof::buffer_virtual_address(outBuf),
		.width = skImageWidth,
		.height = skImageHeight,
	};
	test::dispatch(
		pl,
		push,
		skImageWidth / 8u,
		skImageHeight / 8u
	);
	test::gpu_wait();

	auto const result = test::readback<f32>(outBuf, 0u, pixelCount);

	// every surface has albedo rho and emission Le, so every path returns
	// exactly Le * (1 - rho^B) / (1 - rho) under cosine-weighted sampling:
	// the estimator is zero-variance and a tight per-pixel tolerance holds.
	// rho = 0.8, Le = 0.2, B = 64 -> expected = 1 - 0.8^64 ~= 1.0
	f32 const expected = 1.0f - std::pow(0.8f, 64.0f);
	f32 maxDeviation = 0.0f;
	u32 nanCount = 0u;
	for (u32 i = 0u; i < pixelCount; ++i) {
		if (!std::isfinite(result[i])) {
			nanCount++;
			continue;
		}
		maxDeviation = std::max(
			maxDeviation, std::abs(result[i] - expected)
		);
	}
	CHECK(nanCount == 0u);
	CHECK(maxDeviation < 1e-3f);

	vkof::pipeline_destroy(pl);
	vkof::buffer_destroy(outBuf);
	cornell_scene_destroy(scene);
}

TEST_CASE("cornell: full render") {
	auto const scene = cornell_scene_create(/*closedFront=*/false);
	vkof::acceleration_structure_set_tlas(scene.tlas);

	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "cornell_render.comp",
	});
	REQUIRE(pl.id != 0);

	// higher resolution than the visibility test, for the heatmap eyeball
	// pass; still a 4:3 aspect so no ray runs along the box corner diagonal
	u32 const width = 128u;
	u32 const height = 96u;
	u32 const pixelCount = width * height;
	// light emission, must match skWallEmission in cornell_render.comp
	f32 const skLe = 15.0f;

	auto outBuf = vkof::buffer_create({
		.byteCount = static_cast<u64>(3u * pixelCount) * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	struct Push {
		u64 outVa;
		u32 width;
		u32 height;
	};
	Push const push {
		.outVa = vkof::buffer_virtual_address(outBuf),
		.width = width,
		.height = height,
	};
	test::dispatch(pl, push, width / 8u, height / 8u);
	test::gpu_wait();

	auto const raw = test::readback<f32>(outBuf, 0u, 3u * pixelCount);
	std::vector<f32> const r(
		raw.begin(), raw.begin() + pixelCount
	);
	std::vector<f32> const g(
		raw.begin() + pixelCount, raw.begin() + 2u * pixelCount
	);
	std::vector<f32> const b(
		raw.begin() + 2u * pixelCount, raw.begin() + 3u * pixelCount
	);

	u32 nanCount = 0u;
	f32 minValue = 0.0f;
	for (u32 i = 0u; i < 3u * pixelCount; ++i) {
		if (!std::isfinite(raw[i])) {
			nanCount++;
		} else {
			minValue = std::min(minValue, raw[i]);
		}
	}
	CHECK(nanCount == 0u);
	CHECK(minValue >= 0.0f);

	// classify pixels with the same deterministic camera math the shader
	// uses: a primary ray sees the light iff it crosses the y=0.998 plane
	// inside the light quad. pixels whose crossing lands within 0.02 of the
	// quad edge are left unclassified so cpu/gpu float disagreement at the
	// silhouette cannot flip a check
	auto lightClass = [&](u32 const px, u32 const py) {
		f32 const ndcX = (2.0f * (static_cast<f32>(px) + 0.5f) / width - 1.0f);
		f32 const ndcY = (
			2.0f * (static_cast<f32>(py) + 0.5f) / height - 1.0f
		);
		f32 const dirX = ndcX * 0.4f;
		f32 const dirY = -ndcY * 0.4f;
		if (dirY <= 0.0f) {
			return -1;
		}
		f32 const s = 0.998f / dirY;
		f32 const x = dirX * s;
		f32 const z = -3.0f + s;
		if (std::abs(x) < 0.23f && std::abs(z) < 0.23f) {
			return 1;
		}
		if (std::abs(x) > 0.27f || std::abs(z) > 0.27f) {
			return -1;
		}
		return 0;
	};

	u32 lightPixelCount = 0u;
	f32 lightMin = skLe * 100.0f;
	f32 nonLightMax = 0.0f;
	for (u32 py = 0u; py < height; ++py) {
		for (u32 px = 0u; px < width; ++px) {
			u32 const i = py * width + px;
			f32 const channelMax = std::max(r[i], std::max(g[i], b[i]));
			int const cls = lightClass(px, py);
			if (cls == 1) {
				lightPixelCount++;
				lightMin = std::min(
					lightMin, std::min(r[i], std::min(g[i], b[i]))
				);
			} else if (cls == -1) {
				nonLightMax = std::max(nonLightMax, channelMax);
			}
		}
	}
	// the light quad is visible and reads at least its own emission
	CHECK(lightPixelCount > 0u);
	CHECK(lightMin >= 0.99f * skLe);
	// energy sanity: reflected radiance stays far below the emitter. the
	// analytic bound for the brightest surface (floor under the light) is
	// ~0.3, so 1.0 catches any energy explosion with plenty of margin
	CHECK(nonLightMax < 1.0f);

	// color bleed: the left image third is red-shifted by the red wall,
	// the right third green-shifted by the green wall
	f64 leftR = 0.0, leftG = 0.0, rightR = 0.0, rightG = 0.0;
	for (u32 py = 0u; py < height; ++py) {
		for (u32 px = 0u; px < width / 3u; ++px) {
			leftR += r[py * width + px];
			leftG += g[py * width + px];
		}
		for (u32 px = 2u * width / 3u; px < width; ++px) {
			rightR += r[py * width + px];
			rightG += g[py * width + px];
		}
	}
	CHECK(leftR > 1.2 * leftG);
	CHECK(rightG > 1.2 * rightR);

	// tone-map for the eyeball check: clamp to [0,1], gamma 2.2
	std::vector<f32> pngR(pixelCount), pngG(pixelCount), pngB(pixelCount);
	for (u32 i = 0u; i < pixelCount; ++i) {
		pngR[i] = std::pow(std::clamp(r[i], 0.0f, 1.0f), 1.0f / 2.2f);
		pngG[i] = std::pow(std::clamp(g[i], 0.0f, 1.0f), 1.0f / 2.2f);
		pngB[i] = std::pow(std::clamp(b[i], 0.0f, 1.0f), 1.0f / 2.2f);
	}
	CHECK(test::write_heatmap_png(
		pngR, pngG, pngB, width, height,
		CORNELL_OUTPUT_DIR "cornell-box.png"
	));

	vkof::pipeline_destroy(pl);
	vkof::buffer_destroy(outBuf);
	cornell_scene_destroy(scene);
}

// ---------------------------------------------------------------------------
// openpbr material configs: path-traces the box + interior boxes through the
// full ported material chain (openPbrSampleWo / openPbrEvaluateF), sweeping
// material knobs per config and writing one png each
// ---------------------------------------------------------------------------

namespace {

// mirrors InstanceRecord in cornell_openpbr_render.comp (scalar packing)
struct InstanceRecord {
	u64 vertexVa;
	u64 indexVa;
	f32 baseColor[3];
	f32 baseWeight;
	f32 baseMetalness;
	f32 baseDiffuseRoughness;
	f32 specularWeight;
	f32 specularRoughness;
	f32 specularColor[3];
	f32 specularRoughnessAnisotropy;
	f32 emission[3];
	f32 specularIor;
};
static_assert(sizeof(InstanceRecord) == 80u);

struct SurfaceParams {
	f32 baseColor[3];
	f32 baseMetalness;
	f32 baseDiffuseRoughness;
	f32 specularWeight;
	f32 specularRoughness;
	f32 specularRoughnessAnisotropy;
	f32 specularColor[3];
	f32 specularIor;
};

static SurfaceParams const skSurfWhite {
	.baseColor = { 0.73f, 0.73f, 0.73f },
	.baseMetalness = 0.0f,
	.baseDiffuseRoughness = 0.0f,
	.specularWeight = 0.0f,
	.specularRoughness = 0.3f,
	.specularRoughnessAnisotropy = 0.0f,
	.specularColor = { 1.0f, 1.0f, 1.0f },
	.specularIor = 1.5f,
};

static SurfaceParams surf_diffuse(
	f32 const r, f32 const g, f32 const b, f32 const sigma = 0.0f
) {
	SurfaceParams p = skSurfWhite;
	p.baseColor[0] = r;
	p.baseColor[1] = g;
	p.baseColor[2] = b;
	p.baseDiffuseRoughness = sigma;
	return p;
}

static SurfaceParams surf_metal(
	f32 const r, f32 const g, f32 const b,
	f32 const roughness, f32 const anisotropy = 0.0f
) {
	SurfaceParams p = skSurfWhite;
	p.baseColor[0] = r;
	p.baseColor[1] = g;
	p.baseColor[2] = b;
	p.baseMetalness = 1.0f;
	p.specularWeight = 1.0f;
	p.specularRoughness = roughness;
	p.specularRoughnessAnisotropy = anisotropy;
	return p;
}

static SurfaceParams surf_glossy(
	f32 const r, f32 const g, f32 const b,
	f32 const roughness, f32 const ior = 1.5f
) {
	SurfaceParams p = skSurfWhite;
	p.baseColor[0] = r;
	p.baseColor[1] = g;
	p.baseColor[2] = b;
	p.specularWeight = 1.0f;
	p.specularRoughness = roughness;
	p.specularIor = ior;
	return p;
}

struct MaterialConfig {
	char const * name;
	SurfaceParams tallBox;
	SurfaceParams shortBox;
	// optional overrides for the colored side walls
	bool overrideSideWalls {};
	SurfaceParams leftWall {};
	SurfaceParams rightWall {};
};

static InstanceRecord record_from_surface(
	SurfaceParams const & surf,
	u64 const vertexVa,
	u64 const indexVa,
	f32 const emission = 0.0f
) {
	return InstanceRecord {
		.vertexVa = vertexVa,
		.indexVa = indexVa,
		.baseColor = {
			surf.baseColor[0], surf.baseColor[1], surf.baseColor[2]
		},
		.baseWeight = 1.0f,
		.baseMetalness = surf.baseMetalness,
		.baseDiffuseRoughness = surf.baseDiffuseRoughness,
		.specularWeight = surf.specularWeight,
		.specularRoughness = surf.specularRoughness,
		.specularColor = {
			surf.specularColor[0], surf.specularColor[1], surf.specularColor[2]
		},
		.specularRoughnessAnisotropy = surf.specularRoughnessAnisotropy,
		.emission = { emission, emission, emission },
		.specularIor = surf.specularIor,
	};
}

}

TEST_CASE("cornell: openpbr material configs") {
	auto const scene = cornell_scene_create(
		/*closedFront=*/false, /*withBoxes=*/true
	);
	REQUIRE(scene.instanceCount == skInstanceCountBoxes);
	vkof::acceleration_structure_set_tlas(scene.tlas);

	auto energyTables = ponder::energy_tables_create();

	char const * const includePaths[] = { PONDER_SHADER_DIR };
	auto pl = vkof::pipeline_compute_create({
		.pathCompute = TEST_SHADER_DIR "cornell_openpbr_render.comp",
		.includePaths = srat::slice<char const * const>(includePaths, 1),
	});
	REQUIRE(pl.id != 0);

	u32 const width = 256u;
	u32 const height = 192u;
	u32 const pixelCount = width * height;
	u32 const sampleCount = 384u;
	f32 const skLe = 15.0f;

	auto outBuf = vkof::buffer_create({
		.byteCount = static_cast<u64>(3u * pixelCount) * sizeof(f32),
		.memory = vkof::BufferMemory::DeviceOnly,
	});
	auto sceneBuf = vkof::buffer_create({
		.byteCount = skInstanceCountBoxes * sizeof(InstanceRecord),
		.memory = vkof::BufferMemory::DeviceOnly,
	});

	MaterialConfig const configs[] = {
		{
			.name = "classic-diffuse",
			.tallBox = skSurfWhite,
			.shortBox = skSurfWhite,
		},
		{
			.name = "mirror-silver",
			.tallBox = surf_metal(0.97f, 0.96f, 0.92f, 0.02f),
			.shortBox = surf_metal(0.97f, 0.96f, 0.92f, 0.02f),
		},
		{
			.name = "brushed-gold",
			.tallBox = surf_metal(1.0f, 0.75f, 0.35f, 0.3f),
			.shortBox = surf_metal(1.0f, 0.75f, 0.35f, 0.3f),
		},
		{
			.name = "copper-vs-steel",
			.tallBox = surf_metal(0.95f, 0.54f, 0.43f, 0.12f),
			.shortBox = surf_metal(0.55f, 0.55f, 0.58f, 0.4f),
		},
		{
			.name = "glossy-plastic",
			.tallBox = surf_glossy(0.8f, 0.8f, 0.8f, 0.05f),
			.shortBox = surf_glossy(0.8f, 0.8f, 0.8f, 0.05f),
		},
		{
			.name = "rough-plastic",
			.tallBox = surf_glossy(0.8f, 0.8f, 0.8f, 0.5f),
			.shortBox = surf_glossy(0.8f, 0.8f, 0.8f, 0.5f),
		},
		{
			.name = "cherry-vs-navy",
			.tallBox = surf_glossy(0.65f, 0.05f, 0.04f, 0.03f),
			.shortBox = surf_glossy(0.05f, 0.08f, 0.35f, 0.2f),
		},
		{
			.name = "brushed-aniso",
			.tallBox = surf_metal(0.9f, 0.9f, 0.9f, 0.35f, 0.95f),
			.shortBox = surf_metal(0.9f, 0.9f, 0.9f, 0.35f, 0.0f),
		},
		{
			.name = "eon-sigma-sweep",
			.tallBox = surf_diffuse(0.73f, 0.73f, 0.73f, 0.0f),
			.shortBox = surf_diffuse(0.73f, 0.73f, 0.73f, 1.0f),
		},
		{
			.name = "high-vs-low-ior",
			.tallBox = surf_glossy(0.8f, 0.8f, 0.8f, 0.05f, 2.4f),
			.shortBox = surf_glossy(0.8f, 0.8f, 0.8f, 0.05f, 1.1f),
		},
		{
			.name = "mirror-walls",
			.tallBox = surf_glossy(0.85f, 0.85f, 0.85f, 0.08f),
			.shortBox = surf_glossy(0.85f, 0.85f, 0.85f, 0.08f),
			.overrideSideWalls = true,
			.leftWall = surf_metal(0.95f, 0.35f, 0.3f, 0.03f),
			.rightWall = surf_metal(0.35f, 0.9f, 0.4f, 0.03f),
		},
	};

	for (auto const & config : configs) {
		std::string const configName(config.name);
		CAPTURE(configName);

		SurfaceParams const left = (
			config.overrideSideWalls
			? config.leftWall
			: surf_diffuse(0.63f, 0.065f, 0.05f)
		);
		SurfaceParams const right = (
			config.overrideSideWalls
			? config.rightWall
			: surf_diffuse(0.14f, 0.45f, 0.091f)
		);
		SurfaceParams const surfaces[skInstanceCountBoxes] = {
			skSurfWhite,
			skSurfWhite,
			skSurfWhite,
			left,
			right,
			surf_diffuse(0.78f, 0.78f, 0.78f),
			config.tallBox,
			config.shortBox,
		};
		InstanceRecord records[skInstanceCountBoxes];
		for (u32 inst = 0u; inst < skInstanceCountBoxes; ++inst) {
			records[inst] = record_from_surface(
				surfaces[inst],
				vkof::buffer_virtual_address(scene.vertexBuffers[inst]),
				vkof::buffer_virtual_address(scene.indexBuffers[inst]),
				inst == skWallLight ? skLe : 0.0f
			);
		}
		vkof::buffer_upload({
			.buffer = sceneBuf,
			.byteOffset = 0u,
			.data = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(records), sizeof(records)
			),
		});

		struct Push {
			u64 outVa;
			u64 sceneVa;
			u32 width;
			u32 height;
			u32 sampleCount;
			u32 kullaContyHandle;
		};
		Push const push {
			.outVa = vkof::buffer_virtual_address(outBuf),
			.sceneVa = vkof::buffer_virtual_address(sceneBuf),
			.width = width,
			.height = height,
			.sampleCount = sampleCount,
			.kullaContyHandle = energyTables.kullaContyEnergyHandle,
		};
		test::dispatch(pl, push, width / 8u, height / 8u);
		test::gpu_wait();

		auto const raw = test::readback<f32>(outBuf, 0u, 3u * pixelCount);

		u32 nanCount = 0u;
		f32 minValue = 0.0f;
		f32 maxValue = 0.0f;
		f64 sum = 0.0;
		for (u32 i = 0u; i < 3u * pixelCount; ++i) {
			if (!std::isfinite(raw[i])) {
				nanCount++;
				continue;
			}
			minValue = std::min(minValue, raw[i]);
			maxValue = std::max(maxValue, raw[i]);
			sum += raw[i];
		}
		CHECK(nanCount == 0u);
		CHECK(minValue >= 0.0f);
		// nothing in the scene can meaningfully exceed the emitter: tinted
		// mirrors reflect at most ~Le, so 1.2x is a hard energy ceiling
		CHECK(maxValue < 1.2f * skLe);
		// the image is neither black nor blown out
		f64 const mean = sum / (3.0 * pixelCount);
		CHECK(mean > 0.003);
		CHECK(mean < 2.0);

		std::vector<f32> pngR(pixelCount), pngG(pixelCount), pngB(pixelCount);
		for (u32 i = 0u; i < pixelCount; ++i) {
			pngR[i] = std::pow(
				std::clamp(raw[i], 0.0f, 1.0f), 1.0f / 2.2f
			);
			pngG[i] = std::pow(
				std::clamp(raw[pixelCount + i], 0.0f, 1.0f), 1.0f / 2.2f
			);
			pngB[i] = std::pow(
				std::clamp(raw[2u * pixelCount + i], 0.0f, 1.0f), 1.0f / 2.2f
			);
		}
		std::string const path = (
			std::string(CORNELL_OUTPUT_DIR "cornell-openpbr-")
			+ config.name + ".png"
		);
		CHECK(test::write_heatmap_png(
			pngR, pngG, pngB, width, height, path.c_str()
		));
	}

	vkof::pipeline_destroy(pl);
	vkof::buffer_destroy(sceneBuf);
	vkof::buffer_destroy(outBuf);
	ponder::energy_tables_destroy(energyTables);
	cornell_scene_destroy(scene);
}

}
