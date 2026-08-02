#include <doctest/doctest.h>
#include <vkof/vkof.hpp>
#include <mor/mor.hpp>
#include <mor/mor-shared.h>
#include <ponder/emitter-tables.hpp>
#include <ponder/energy-tables.hpp>
#include <ponder/environment-tables.hpp>
#include <ponder/debug-path.hpp>
#include <ponder/wavefront.hpp>
#include <ponder/zeltner-tables.hpp>
#include <srat/camera.hpp>
#include <srat/core-math.hpp>
#include "util.hpp"

#include "viewer/shaders/resolve_pc.h"

#include <tinyexr.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// tier 3 of plan-nee.md: emissive-mesh NEE through the *production* path
// tracer. tiers 0-2 prove the table, the sampler/pdf pair and the estimator
// arithmetic in isolation; none of them touch the integration -- the path
// state threaded through utilIrradianceWalk, the delta-bounce exclusion, or
// the backward hit lookup against real traced geometry. That is what fails
// silently, and that is what this file renders.
//
// Two things make the comparison trustworthy:
//
//  1. It does not go through pt-accumulate.comp. That pass averages
//     sum(rgb*a)/sum(a) over light-carrying samples only, so NEE-on and
//     NEE-off normalize over different sample populations and their means are
//     not comparable at all (measured: a 55% apparent difference with no bug
//     present). emitter_nee_scene.comp averages radianceSum/sampleCount flat.
//  2. The scenes are built from hand-authored mor buffers rather than a glTF,
//     so the geometry is exactly a flat quad emitter over a flat diffuse
//     floor: no curved emissive geometry to self-shadow, no transmission, no
//     textures. Every candidate cause of a discrepancy is either removed or
//     switched on deliberately.
//
// The controlled scene also has a closed form. A Lambertian floor under a
// uniform-radiance quad has an analytic single-bounce irradiance (Lambert's
// polygon formula), so the render is checked against a number, not only
// against its own other configuration.
// ---------------------------------------------------------------------------

namespace {

// one long-lived handle per table, same reasoning as test-env-mis.cpp: the
// bindless-slot churn from per-case create/destroy made unrelated cases fail
// depending on ordering
u32 shared_kulla_conty_handle() {
	static ponder::EnergyTables const tables = ponder::energy_tables_create();
	return tables.kullaContyEnergyHandle;
}

u32 shared_zeltner_handle() {
	static ponder::ZeltnerTables const tables = ponder::zeltner_tables_create();
	return tables.zeltnerLtcParamHandle;
}

// mirrors GpuEmitterScenePC in emitter_nee_scene.comp
struct GpuEmitterScenePC {
	u64 outVa;
	u32 width;
	u32 height;
	u32 sampleCount;
	u32 propagationDepth;
	u32 kullaContyEnergyHandle;
	u32 zeltnerLtcParamHandle;
	u32 seedSalt;
	u32 pad0;
};

// -- material construction -------------------------------------------------

GpuMorMaterialComponent1 comp1(f32 const v) {
	return GpuMorMaterialComponent1 {
		.r = v, .texture = 0u, .swizzle = 0, .uvTransform = 0u
	};
}

GpuMorMaterialComponent3 comp3(f32v3 const v) {
	return GpuMorMaterialComponent3 { .rgb = v, .texture = 0u, .uvTransform = 0u };
}

// every field set explicitly: openPbrLoadMaterialDeriv reads all of them, and
// an uninitialised weight here would read as a lobe that should not exist
GpuMorMaterial make_material(
	f32v3 const baseColor,
	f32v3 const emissionColor,
	f32 const emissionLuminance
) {
	GpuMorMaterial m {};
	m.baseWeight = comp1(1.0f);
	m.baseColor = comp3(baseColor);
	m.baseMetalness = comp1(0.0f);
	m.baseDiffuseRoughness = comp1(0.0f);
	m.specularWeight = comp1(0.0f);
	m.specularColor = comp3({ 1.0f, 1.0f, 1.0f });
	m.specularRoughness = comp1(0.5f);
	m.specularRoughnessAnisotropy = comp1(0.0f);
	m.specularRoughnessAnisotropyRotation = comp1(0.0f);
	m.specularIor = comp1(1.5f);
	m.transmissionWeight = comp1(0.0f);
	m.transmissionColor = comp3({ 1.0f, 1.0f, 1.0f });
	m.transmissionDepth = comp1(0.0f);
	m.transmissionScatter = comp3({ 0.0f, 0.0f, 0.0f });
	m.transmissionScatterAnisotropy = comp1(0.0f);
	m.transmissionDispersionScale = comp1(0.0f);
	m.transmissionDispersionAbbeNumber = comp1(20.0f);
	m.subsurfaceWeight = comp1(0.0f);
	m.subsurfaceColor = comp3({ 0.8f, 0.8f, 0.8f });
	m.subsurfaceRadius = comp1(1.0f);
	m.subsurfaceRadiusScale = comp3({ 1.0f, 1.0f, 1.0f });
	m.subsurfaceScatterAnisotropy = comp1(0.0f);
	m.coatWeight = comp1(0.0f);
	m.coatColor = comp3({ 1.0f, 1.0f, 1.0f });
	m.coatRoughness = comp1(0.0f);
	m.coatRoughnessAnisotropy = comp1(0.0f);
	m.coatIor = comp1(1.6f);
	m.coatDarkening = comp1(1.0f);
	m.fuzzWeight = comp1(0.0f);
	m.fuzzColor = comp3({ 1.0f, 1.0f, 1.0f });
	m.fuzzRoughness = comp1(0.5f);
	m.emissionLuminance = comp1(emissionLuminance);
	m.emissionColor = comp3(emissionColor);
	m.thinFilmWeight = comp1(0.0f);
	m.thinFilmThickness = comp1(0.5f);
	m.thinFilmIor = comp1(1.4f);
	m.geometryOpacity = comp1(1.0f);
	m.geometryThinWalled = comp1(0.0f);
	m.geometryNormalTexture = 0u;
	m.geometryNormalUvTransform = 0u;
	m.geometryTangentTexture = 0u;
	m.geometryTangentUvTransform = 0u;
	m.geometryCoatNormalTexture = 0u;
	m.geometryCoatNormalUvTransform = 0u;
	m.geometryCoatTangentTexture = 0u;
	m.geometryCoatTangentUvTransform = 0u;
	m.alphaCutoff = 0.5f;
	m.alphaMode = MOR_ALPHA_MODE_OPAQUE;
	return m;
}

// -- hand-built mor scene --------------------------------------------------

// a scene the production path tracer can resolve materials through. mor's
// upload path is bypassed on purpose: these are the exact buffers
// utilMeshAttributeDataFromIndices indexes (positions, attributes,
// flatIndices, flatMeshlets, meshlets, materials, uvTransforms), built so a
// test can place a single triangle wherever it wants
struct MorScene {
	std::vector<vkof::Buffer> buffers;
	vkof::Buffer models;
	vkof::Buffer emitterLights;
	u32 emitterLightCount = 0u;
	// host copy of the same list, so a test can append the environment entry
	// (or re-weight it) and upload its own variant -- what tiers 2.4/2.5 need
	std::vector<GpuEmitterLight> lightsCpu;
	vkof::AccelerationStructureBlas blas {};
	vkof::AccelerationStructureTlas tlas {};

	void destroy() {
		if (tlas.id != 0u) { vkof::tlas_destroy(tlas); }
		if (blas.id != 0u) { vkof::blas_destroy(blas); }
		if (emitterLights.id != 0u) { vkof::buffer_destroy(emitterLights); }
		if (models.id != 0u) { vkof::buffer_destroy(models); }
		for (vkof::Buffer & b : buffers) { vkof::buffer_destroy(b); }
	}
};

struct MeshTri {
	f32v3 a;
	f32v3 b;
	f32v3 c;
	// index into the scene's material list
	u32 materialIndex;
	// per-corner uv. zero everywhere unless a test needs a texture to vary
	// across the surface, which only the emissive-texture case does
	f32v2 uvA { 0.0f, 0.0f };
	f32v2 uvB { 0.0f, 0.0f };
	f32v2 uvC { 0.0f, 0.0f };
};

// one meshlet per material run, matching how mor groups triangles; the
// emitter build reconstructs primitiveIndexBase from these triangle counts,
// which is the same assumption the real loader satisfies
MorScene build_scene(
	std::vector<MeshTri> const & tris,
	std::vector<GpuMorMaterial> const & materials,
	// the emitter list's cap. lowering it clips the list exactly as a
	// pathological scene would, and -- unlike overriding the uploaded buffer --
	// keeps each model's emitterLightOffset/Count describing the list the
	// shader actually reads, which the backward lookup indexes into
	u32 const maxLights = 4096u
) {
	MorScene scene {};

	std::vector<f32v3> positions;
	std::vector<GpuMorVertexAttribute> attributes;
	std::vector<u32> flatIndices;
	std::vector<u32> flatMeshlets;
	std::vector<GpuMorMeshlet> meshlets;

	// group consecutive triangles sharing a material into one meshlet
	u32 t = 0u;
	while (t < (u32)tris.size()) {
		u32 const materialIndex = tris[t].materialIndex;
		u32 const meshletIndex = (u32)meshlets.size();
		u32 const triangleOffset = t;
		u32 count = 0u;
		while (
			t < (u32)tris.size() && tris[t].materialIndex == materialIndex
		) {
			MeshTri const & tri = tris[t];
			u32 const base = (u32)positions.size();
			f32v3 const geoNormal = (
				f32v3_normalize(f32v3_cross(tri.b - tri.a, tri.c - tri.a))
			);
			f32v3 const corners[3] = { tri.a, tri.b, tri.c };
			f32v2 const cornerUvs[3] = { tri.uvA, tri.uvB, tri.uvC };
			for (u32 c = 0u; c < 3u; ++c) {
				positions.emplace_back(corners[c]);
				attributes.emplace_back(GpuMorVertexAttribute {
					.normal = geoNormal,
					.uv = cornerUvs[c],
					.tangent = { 1.0f, 0.0f, 0.0f, 1.0f },
				});
			}
			flatIndices.emplace_back(base + 0u);
			flatIndices.emplace_back(base + 1u);
			flatIndices.emplace_back(base + 2u);
			flatMeshlets.emplace_back(meshletIndex);
			++count;
			++t;
		}
		meshlets.emplace_back(GpuMorMeshlet {
			.vertexOffset = 0u,
			.vertexCount = 0u,
			.triangleOffset = triangleOffset * 3u,
			.triangleCount = count,
			.instanceIndex = 0u,
			.materialIndex = materialIndex,
		});
	}

	// index 0 must be the identity transform (mor's convention)
	std::vector<GpuMorUvTransform> const uvTransforms {
		GpuMorUvTransform {
			.basisU = { 1.0f, 0.0f },
			.basisV = { 0.0f, 1.0f },
			.offset = { 0.0f, 0.0f },
		},
	};

	auto upload = [&](void const * data, u64 const bytes) {
		vkof::Buffer const b = test::upload_bytes(data, bytes);
		scene.buffers.emplace_back(b);
		return vkof::buffer_virtual_address(b);
	};

	u64 const positionsVa = (
		upload(positions.data(), positions.size() * sizeof(f32v3))
	);
	u64 const attributesVa = (
		upload(
			attributes.data(), attributes.size() * sizeof(GpuMorVertexAttribute)
		)
	);
	u64 const flatIndicesVa = (
		upload(flatIndices.data(), flatIndices.size() * sizeof(u32))
	);
	u64 const flatMeshletsVa = (
		upload(flatMeshlets.data(), flatMeshlets.size() * sizeof(u32))
	);
	u64 const meshletsVa = (
		upload(meshlets.data(), meshlets.size() * sizeof(GpuMorMeshlet))
	);
	u64 const materialsVa = (
		upload(materials.data(), materials.size() * sizeof(GpuMorMaterial))
	);
	u64 const uvTransformsVa = (
		upload(uvTransforms.data(), uvTransforms.size() * sizeof(GpuMorUvTransform))
	);

	// -- emitter list, built the same way the viewer builds it per frame
	std::vector<GpuEmitterLight> lights;
	{
		std::vector<ponder::EmitterMeshletInput> inputs;
		u32 primitiveIndexBase = 0u;
		for (GpuMorMeshlet const & m : meshlets) {
			u32 const thisBase = primitiveIndexBase;
			primitiveIndexBase += m.triangleCount;
			GpuMorMaterial const & mat = materials[m.materialIndex];
			if (mat.emissionLuminance.r <= 0.0f) { continue; }
			f32 area = 0.0f;
			for (u32 i = 0u; i < m.triangleCount; ++i) {
				MeshTri const & tri = tris[thisBase + i];
				area += 0.5f * f32v3_length(
					f32v3_cross(tri.b - tri.a, tri.c - tri.a)
				);
			}
			inputs.emplace_back(ponder::EmitterMeshletInput {
				.primitiveIndexBase = thisBase,
				.triangleCount = m.triangleCount,
				.objectArea = area,
				.emissionColor = mat.emissionColor.rgb,
				.emissionLuminance = mat.emissionLuminance.r,
			});
		}
		(void)ponder::emitter_lights_append(
			lights,
			srat::slice<ponder::EmitterMeshletInput const>(
				inputs.data(), inputs.size()
			),
			f32m44_identity(), /*modelDrawIndex=*/0u, maxLights
		);
		(void)ponder::emitter_lights_finalize(lights);
	}
	scene.emitterLightCount = (u32)lights.size();
	scene.lightsCpu = lights;
	if (!lights.empty()) {
		scene.emitterLights = (
			test::upload_bytes(
				lights.data(), lights.size() * sizeof(GpuEmitterLight)
			)
		);
	}

	GpuResolveModelIndirect model {};
	model.meshlets = meshletsVa;
	model.materials = materialsVa;
	model.uvTransforms = uvTransformsVa;
	model.positions = positionsVa;
	model.attributes = attributesVa;
	model.flatIndices = flatIndicesVa;
	model.flatMeshlets = flatMeshletsVa;
	model.modelMatrix = f32m44_identity();
	model.emitterLightOffset = 0u;
	model.emitterLightCount = scene.emitterLightCount;
	scene.models = test::upload_bytes(&model, sizeof(model));

	// -- blas/tlas over the same positions + flat indices the shader reads,
	// so a ray hit's primitiveIndex indexes exactly those arrays
	scene.blas = vkof::blas_create({
		.positionVa = positionsVa,
		.vertexCount = (u32)positions.size(),
		.indexVa = flatIndicesVa,
		.triangleCount = (u32)tris.size(),
		.isOpaque = true,
	});
	REQUIRE(scene.blas.id != 0u);
	scene.tlas = vkof::tlas_create({ .maxInstances = 1u });
	REQUIRE(scene.tlas.id != 0u);

	vkof::TlasInstance const instance {
		.blas = scene.blas,
		.transform = f32m44_identity(),
		.instanceCustomIndex = 0u,
		.rayMask = 0xFFu,
	};
	vkof::RenderNode const node = (
		vkof::render_node_create({ .queue = vkof::CommandQueue::compute })
	);
	vkof::render_node_callback({
		.node = node,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			vkof::tlas_build(
				cmd, scene.tlas,
				srat::slice<vkof::TlasInstance const>(&instance, 1u)
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

// -- render ----------------------------------------------------------------

struct RenderConfig {
	u32 width = 64u;
	u32 height = 64u;
	u32 sampleCount = 512u;
	u32 propagationDepth = 8u;
	bool meshNee = true;
	u32 seedSalt = 0u;
	f32 fireflyClamp = 0.0f;
	// compiles util-raytrace.glsl with EMITTER_NEE_DEBUG and reads back the
	// nee counters; off by default so the ordinary renders stay untouched
	bool collectNeeCounters = false;
	// -- environment. 0 keeps ENV_MODE_BLACK, i.e. the emissive mesh is the
	// only light in the scene, which is what every mesh-only case wants
	u64 envMapHandlesVa = 0u;
	f32 envIntensity = 1.0f;
	bool envNee = true;
	// replaces the scene's own mesh-only nee list. this is how the
	// environment entry gets into the list at all, and how the power-split
	// sweep forces a pick share without touching the scene
	vkof::Buffer lightsOverride {};
	u32 lightsOverrideCount = 0u;
	// index of the environment entry within that list; anything >= the count
	// means "not in the list", which is what the shader tests
	u32 envLightIndex = 0xFFFFFFFFu;
	// -- homogeneous fog. sigma_t per world unit; > 0 turns the whole scene
	// into a scattering medium, which is the only way to reach the vdb-medium
	// nee site without authoring a nanovdb grid -- the two share one code path
	f32 fogThickness = 0.0f;
	f32v3 fogAlbedo { 0.0f, 0.0f, 0.0f };
	f32 fogDistanceMax = 0.0f;
};

struct RenderResult {
	std::vector<f32> rgb;
	f64 mean;
	u32 nonFiniteCount;
	// slots: 0 attempts, 1 sampler valid, 2 passed f/cos, 3 occluded,
	// 4 contributed
	std::vector<u32> neeCounters;
};

// templated on the scene type: it only ever reaches for emitterLights,
// emitterLightCount, models and tlas, which the instanced builder below
// provides under the same names
template <typename SceneT>
RenderResult render_scene(
	SceneT const & scene,
	srat::CameraOrbit const & camera,
	RenderConfig const & cfg
) {
	char const * const includePaths[] = {
		PONDER_SHADER_DIR, MOR_INCLUDE_DIR, APP_SHADER_DIR, TEST_SHADER_DIR,
	};
	char const * const debugDefines[] = { "EMITTER_NEE_DEBUG" };
	vkof::Pipeline const pipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = TEST_SHADER_DIR "slang/entry/emitter_nee_scene.slang",
			.defines = srat::slice<char const * const>(
				debugDefines, cfg.collectNeeCounters ? 1u : 0u
			),
			.includePaths = srat::slice<char const * const>(includePaths, 4u),
		})
	);
	REQUIRE(pipeline.id != 0u);
	std::vector<u32> const zeroCounters(7u, 0u);
	vkof::Buffer neeDebug = (
		test::upload_bytes(zeroCounters.data(), zeroCounters.size() * sizeof(u32))
	);

	u32 const pixelCount = cfg.width * cfg.height;
	vkof::Buffer out = test::make_buffer_u32(pixelCount * 3u);
	vkof::Buffer nanCounter = test::make_buffer_u32(1u);

	// 3 == ENV_MODE_HDRMAP, 2 == ENV_MODE_BLACK; without a map the emissive
	// mesh must be the only light in the scene, or an env contribution would
	// swamp what is being measured
	bool const hasEnv = cfg.envMapHandlesVa != 0u;
	bool const overrideLights = cfg.lightsOverride.id != 0u;

	GpuGlobalExtended extended {};
	extended.envIntensity = hasEnv ? cfg.envIntensity : 0.0f;
	extended.renderWidth = cfg.width;
	extended.renderHeight = cfg.height;
	extended.envMode = hasEnv ? 3 : 2;
	extended.antialias = 0u;
	extended.frameMode = 2;
	extended.nanProbeCounterVa = vkof::buffer_virtual_address(nanCounter);
	extended.probePixel = { -1, -1 };
	extended.envMap = cfg.envMapHandlesVa;
	extended.envNeeEnabled = (hasEnv && cfg.envNee) ? 1u : 0u;
	extended.emitterLights = (
		overrideLights
			? vkof::buffer_virtual_address(cfg.lightsOverride)
			: (
				scene.emitterLights.id != 0u
					? vkof::buffer_virtual_address(scene.emitterLights) : 0u
			)
	);
	extended.emitterLightCount = (
		overrideLights
			? cfg.lightsOverrideCount
			: (cfg.meshNee ? scene.emitterLightCount : 0u)
	);
	extended.meshNeeEnabled = cfg.meshNee ? 1u : 0u;
	extended.envLightIndex = cfg.envLightIndex;
	extended.emitterNeeDebug = (
		cfg.collectNeeCounters ? vkof::buffer_virtual_address(neeDebug) : 0u
	);
	extended.envMapOnlyDepth = 0u;
	extended.fireflyClampLuminance = cfg.fireflyClamp;
	extended.envBackgroundIntensity = 1.0f;
	extended.vdb = 0u;
	extended.vdbWorldToLocal = f32m44_identity();
	extended.vdbPhaseAnisotropyOverride = -2.0f;
	extended.fogThickness = cfg.fogThickness;
	extended.fogAlbedo = cfg.fogAlbedo;
	extended.fogDistanceMax = cfg.fogDistanceMax;

	vkof::Buffer extendedBuffer = (
		test::upload_bytes(&extended, sizeof(extended))
	);

	vkof::acceleration_structure_set_tlas(scene.tlas);

	GpuGlobalPc const globalPc {
		.time = 0.0f,
		.cameraPos = srat::camera_orbit_eye(camera),
		.exposure = 1.0f,
		.pad0 = 0.0f,
		.viewProj = (
			srat::camera_orbit_proj(camera) * srat::camera_orbit_view(camera)
		),
		.extended = vkof::buffer_virtual_address(extendedBuffer),
		.models = vkof::buffer_virtual_address(scene.models),
		.pad1 = 0u,
		.pad2 = 0u,
		.pad3 = 0u,
	};
	GpuEmitterScenePC const push {
		.outVa = vkof::buffer_virtual_address(out),
		.width = cfg.width,
		.height = cfg.height,
		.sampleCount = cfg.sampleCount,
		.propagationDepth = cfg.propagationDepth,
		.kullaContyEnergyHandle = shared_kulla_conty_handle(),
		.zeltnerLtcParamHandle = shared_zeltner_handle(),
		.seedSalt = cfg.seedSalt,
		.pad0 = 0u,
	};

	test::dispatch(
		pipeline, push, (cfg.width + 7u) / 8u, (cfg.height + 7u) / 8u, 1u,
		srat::slice_as_bytes(globalPc)
	);
	test::gpu_wait();

	RenderResult result {};
	result.rgb = test::readback<f32>(out, 0, pixelCount * 3u);
	f64 sum = 0.0;
	for (f32 const v : result.rgb) {
		if (!std::isfinite(v)) { result.nonFiniteCount++; continue; }
		sum += (f64)v;
	}
	result.mean = sum / (f64)result.rgb.size();

	if (cfg.collectNeeCounters) {
		result.neeCounters = test::readback<u32>(neeDebug, 0, 7u);
	}
	vkof::buffer_destroy(neeDebug);
	vkof::buffer_destroy(extendedBuffer);
	vkof::buffer_destroy(nanCounter);
	vkof::buffer_destroy(out);
	vkof::pipeline_destroy(pipeline);
	return result;
}

// mean of n independent renders plus the standard error of that mean. the
// half-spread of a *pair* is a one-degree-of-freedom estimate of sigma, so it
// lands anywhere from ~0 to ~3x the truth -- fine for gating a percent-scale
// discrepancy, useless for gating a sub-0.1% one, where the gate then flakes
// on the sigma rather than on the mean
struct MeanAndError {
	f64 mean;
	f64 error;
	u32 nonFiniteCount;
};

template <typename SceneT>
MeanAndError render_mean(
	SceneT const & scene,
	srat::CameraOrbit const & camera,
	RenderConfig const & cfg,
	u32 const realizations
) {
	REQUIRE(realizations >= 2u);
	std::vector<f64> samples;
	MeanAndError out { 0.0, 0.0, 0u };
	for (u32 r = 0u; r < realizations; ++r) {
		RenderConfig r_cfg = cfg;
		r_cfg.seedSalt = cfg.seedSalt + 13u * r;
		RenderResult const res = render_scene(scene, camera, r_cfg);
		out.nonFiniteCount += res.nonFiniteCount;
		samples.emplace_back(res.mean);
	}
	f64 sum = 0.0;
	for (f64 const v : samples) { sum += v; }
	out.mean = sum / (f64)realizations;
	f64 sumSq = 0.0;
	for (f64 const v : samples) { sumSq += (v - out.mean) * (v - out.mean); }
	out.error = (
		std::sqrt(sumSq / (f64)(realizations - 1u)) / std::sqrt((f64)realizations)
	);
	return out;
}

// per-pixel comparison between two renders of the same scene
struct ImageDiff {
	f64 meanA;
	f64 meanB;
	f64 meanAbsDiff;
	f64 maxAbsDiff;
	f64 relativeMeanDiff;
};

ImageDiff compare(RenderResult const & a, RenderResult const & b) {
	REQUIRE(a.rgb.size() == b.rgb.size());
	f64 sumDiff = 0.0;
	f64 maxDiff = 0.0;
	for (size_t i = 0u; i < a.rgb.size(); ++i) {
		f64 const d = std::abs((f64)a.rgb[i] - (f64)b.rgb[i]);
		sumDiff += d;
		maxDiff = std::max(maxDiff, d);
	}
	return {
		a.mean, b.mean, sumDiff / (f64)a.rgb.size(), maxDiff,
		b.mean > 0.0 ? std::abs(a.mean - b.mean) / b.mean : 0.0,
	};
}

// -- scenes ----------------------------------------------------------------

// a quad in the y = height plane, split into two triangles
void push_quad(
	std::vector<MeshTri> & out,
	f32 const halfSize,
	f32 const height,
	u32 const materialIndex,
	bool const faceDown
) {
	f32v3 const v00 { -halfSize, height, -halfSize };
	f32v3 const v10 { halfSize, height, -halfSize };
	f32v3 const v11 { halfSize, height, halfSize };
	f32v3 const v01 { -halfSize, height, halfSize };
	// uv follows the corner layout, so a texture maps once across the quad
	f32v2 const t00 { 0.0f, 0.0f };
	f32v2 const t10 { 1.0f, 0.0f };
	f32v2 const t11 { 1.0f, 1.0f };
	f32v2 const t01 { 0.0f, 1.0f };
	if (faceDown) {
		out.emplace_back(MeshTri { v00, v11, v10, materialIndex, t00, t11, t10 });
		out.emplace_back(MeshTri { v00, v01, v11, materialIndex, t00, t01, t11 });
	} else {
		out.emplace_back(MeshTri { v00, v10, v11, materialIndex, t00, t10, t11 });
		out.emplace_back(MeshTri { v00, v11, v01, materialIndex, t00, t11, t01 });
	}
}

// floor at y = 0 (material 0, diffuse) with an emissive quad above it
// (material 1). the emitter is deliberately smaller than the floor so most
// floor pixels see it at an angle
MorScene build_floor_and_emitter(
	f32 const emitterHalfSize,
	f32 const emitterHeight,
	f32 const emissionLuminance
) {
	std::vector<GpuMorMaterial> const materials {
		make_material({ 0.8f, 0.8f, 0.8f }, { 0.0f, 0.0f, 0.0f }, 0.0f),
		make_material(
			{ 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, emissionLuminance
		),
	};
	std::vector<MeshTri> tris;
	push_quad(tris, 8.0f, 0.0f, /*materialIndex=*/0u, /*faceDown=*/false);
	push_quad(tris, emitterHalfSize, emitterHeight, 1u, /*faceDown=*/true);
	return build_scene(tris, materials);
}

// an axis-aligned box as 12 outward-facing triangles in one material run. the
// subsurface walk needs a closed volume to random-walk inside and refract out
// of; a flat quad has no interior for it to enter
void push_box(
	std::vector<MeshTri> & out,
	f32v3 const center,
	f32 const halfExtent,
	u32 const materialIndex
) {
	f32 const h = halfExtent;
	f32v3 const corner[8] = {
		center + f32v3 { -h, -h, -h },
		center + f32v3 { h, -h, -h },
		center + f32v3 { h, h, -h },
		center + f32v3 { -h, h, -h },
		center + f32v3 { -h, -h, h },
		center + f32v3 { h, -h, h },
		center + f32v3 { h, h, h },
		center + f32v3 { -h, h, h },
	};
	// each quad wound counter-clockwise seen from outside the box
	u32 const face[6][4] = {
		{ 0u, 3u, 2u, 1u },
		{ 4u, 5u, 6u, 7u },
		{ 0u, 1u, 5u, 4u },
		{ 3u, 7u, 6u, 2u },
		{ 0u, 4u, 7u, 3u },
		{ 1u, 2u, 6u, 5u },
	};
	for (u32 f = 0u; f < 6u; ++f) {
		out.emplace_back(MeshTri {
			corner[face[f][0]], corner[face[f][1]], corner[face[f][2]],
			materialIndex,
		});
		out.emplace_back(MeshTri {
			corner[face[f][0]], corner[face[f][2]], corner[face[f][3]],
			materialIndex,
		});
	}
}

// a purely subsurface material: the walk replaces the diffuse response
// entirely, so every path entering it leaves through an exit vertex, which is
// the site tier 3.4 measures
GpuMorMaterial make_subsurface_material(
	f32v3 const subsurfaceColor,
	f32 const subsurfaceRadius,
	f32 const exitRoughness
) {
	GpuMorMaterial m = make_material({ 0.5f, 0.5f, 0.5f }, { 0.0f, 0.0f, 0.0f }, 0.0f);
	// a real dielectric interface for the walk to refract out through.
	// specularWeight 0 drives openPbrEffectiveIor to exactly 1.0, i.e. no
	// interface at all, which the exit's rough microfacet btdf is degenerate at
	m.specularWeight = comp1(1.0f);
	// the exit interface's slopes. this is the axis the tir-gating defect
	// scaled with, so the test sweeps it rather than picking one value
	m.specularRoughness = comp1(exitRoughness);
	m.subsurfaceWeight = comp1(1.0f);
	m.subsurfaceColor = comp3(subsurfaceColor);
	m.subsurfaceRadius = comp1(subsurfaceRadius);
	m.subsurfaceRadiusScale = comp3({ 1.0f, 1.0f, 1.0f });
	return m;
}

// looks down at the floor from an angle, so the frame is mostly lit floor
// rather than emitter or void
srat::CameraOrbit overhead_camera(f32 const width, f32 const height) {
	srat::CameraOrbit cam {};
	cam.target = { 0.0f, 0.5f, 0.0f };
	cam.distance = 7.0f;
	cam.azimuth = 0.6f;
	cam.elevation = 0.55f;
	cam.fovY = 0.9f;
	cam.aspect = width / height;
	cam.near = 0.05f;
	cam.far = 200.0f;
	return cam;
}

// -- environment map -------------------------------------------------------

// mirrors util-environment-map.glsl's EnvironmentMapHandles field order
struct GpuEnvironmentMapHandles {
	u32 radiance;
	u32 pdf;
	u32 aliasProb;
	u32 aliasIndex;
};

struct EnvScene {
	ponder::EnvironmentTables tables {};
	vkof::Buffer handles {};

	u64 va() const {
		return handles.id != 0u ? vkof::buffer_virtual_address(handles) : 0u;
	}

	void destroy() {
		if (handles.id != 0u) { vkof::buffer_destroy(handles); }
		if (tables.width != 0u) { ponder::environment_tables_destroy(tables); }
	}
};

// a smooth, everywhere-positive sky, written out and loaded back through the
// production exr path so environment_tables_create runs end to end. no sun on
// purpose: tiers 2.4/2.5 measure the bookkeeping *between* two light
// techniques, and a high-contrast map would bury that under the variance of
// the env technique itself (test-env-mis.cpp is where contrast is swept)
EnvScene make_env_sky(char const * const path) {
	constexpr i32 skWidth = 64;
	constexpr i32 skHeight = 32;
	std::vector<f32> pixels((size_t)skWidth * skHeight * 4);
	for (i32 y = 0; y < skHeight; ++y) {
		for (i32 x = 0; x < skWidth; ++x) {
			f32 const u = (f32)x / (f32)skWidth;
			f32 const v = (f32)y / (f32)skHeight;
			size_t const i = ((size_t)y * skWidth + x) * 4;
			// mildly coloured and mildly varying, so a dropped channel or a
			// flipped axis cannot hide behind a flat grey
			pixels[i + 0] = 0.30f + 0.10f * u;
			pixels[i + 1] = 0.34f + 0.08f * v;
			pixels[i + 2] = 0.40f + 0.06f * (1.0f - u);
			pixels[i + 3] = 1.0f;
		}
	}
	char const * err = nullptr;
	i32 const ret = (
		SaveEXR(pixels.data(), skWidth, skHeight, 4, /*save_as_fp16=*/0, path, &err)
	);
	if (ret != TINYEXR_SUCCESS && err != nullptr) {
		MESSAGE("SaveEXR failed: ", err);
		FreeEXRErrorMessage(err);
	}
	REQUIRE(ret == TINYEXR_SUCCESS);

	EnvScene env {};
	env.tables = ponder::environment_tables_create(path);
	REQUIRE(env.tables.width != 0u);
	GpuEnvironmentMapHandles const handles {
		.radiance = env.tables.radianceHandle,
		.pdf = env.tables.pdfHandle,
		.aliasProb = env.tables.aliasProbHandle,
		.aliasIndex = env.tables.aliasIndexHandle,
	};
	env.handles = test::upload_bytes(&handles, sizeof(handles));
	return env;
}

// the frame's combined light list: the scene's mesh entries plus one
// environment entry, re-finalized so the cdf spans both. envPower <= 0 leaves
// the environment out, which is how the mesh-only leg of the linearity test
// is built
struct CombinedLightList {
	vkof::Buffer buffer {};
	u32 count = 0u;
	u32 envLightIndex = 0xFFFFFFFFu;

	void destroy() {
		if (buffer.id != 0u) { vkof::buffer_destroy(buffer); }
	}
};

CombinedLightList make_light_list(
	std::vector<GpuEmitterLight> const & meshLights,
	f32 const envPower
) {
	std::vector<GpuEmitterLight> lights = meshLights;
	u32 const envIndex = (
		ponder::emitter_lights_append_environment(lights, envPower, 4096u)
	);
	CombinedLightList out {};
	if (lights.empty()) {
		return out;
	}
	(void)ponder::emitter_lights_finalize(lights);
	out.count = (u32)lights.size();
	out.envLightIndex = envIndex;
	out.buffer = (
		test::upload_bytes(lights.data(), lights.size() * sizeof(GpuEmitterLight))
	);
	return out;
}

// total power of the mesh half of the list, so a test can solve for the
// environment power that yields a chosen pick share
f64 mesh_power_total(std::vector<GpuEmitterLight> const & meshLights) {
	f64 total = 0.0;
	for (GpuEmitterLight const & l : meshLights) {
		total += (f64)l.power;
	}
	return total;
}

} // namespace

TEST_CASE("emitter nee scene: on and off converge to the same image") {
	// the headline test. a flat quad emitter over a flat diffuse floor --
	// no curved geometry, no transmission, no textures, no environment -- so
	// nothing but the nee integration differs between the two renders.
	//
	// the tolerance is *measured*, not picked: each config is rendered twice
	// with decorrelated rng, so the spread between its own two realizations
	// estimates the standard error of its mean. a fixed percentage would
	// either hide a real bias (too loose) or fail on noise (too tight), and
	// which of those it does would change with sample count.
	MorScene scene = build_floor_and_emitter(1.0f, 3.0f, 4.0f);
	REQUIRE(scene.emitterLightCount > 0u);

	RenderConfig cfg {};
	cfg.sampleCount = test::statistical_samples(2048u);
	srat::CameraOrbit const cam = overhead_camera((f32)cfg.width, (f32)cfg.height);

	auto const renderPair = [&](bool const meshNee) {
		RenderConfig a = cfg;
		a.meshNee = meshNee;
		a.seedSalt = 11u;
		RenderConfig b = cfg;
		b.meshNee = meshNee;
		b.seedSalt = 29u;
		return std::pair<RenderResult, RenderResult> {
			render_scene(scene, cam, a), render_scene(scene, cam, b)
		};
	};

	auto const [on1, on2] = renderPair(true);
	auto const [off1, off2] = renderPair(false);

	CHECK(on1.nonFiniteCount == 0u);
	CHECK(off1.nonFiniteCount == 0u);

	f64 const meanOn = 0.5 * (on1.mean + on2.mean);
	f64 const meanOff = 0.5 * (off1.mean + off2.mean);
	REQUIRE(meanOff > 0.0);

	// half the spread between two independent realizations estimates one
	// realization's standard error; the two configs' errors add in quadrature
	f64 const sigmaOn = 0.5 * std::abs(on1.mean - on2.mean);
	f64 const sigmaOff = 0.5 * std::abs(off1.mean - off2.mean);
	f64 const sigma = std::sqrt(sigmaOn * sigmaOn + sigmaOff * sigmaOff);
	f64 const delta = std::abs(meanOn - meanOff);

	MESSAGE(
		"nee-on ", meanOn, " vs nee-off ", meanOff,
		"  delta ", delta / meanOff * 100.0, "%",
		"  sigma ", sigma / meanOff * 100.0, "%",
		"  = ", delta / std::max(sigma, 1e-12), " sigma"
	);
	CAPTURE(meanOn);
	CAPTURE(meanOff);
	CAPTURE(sigma);
	// 4 sigma: tight enough that a systematic loss of even a fraction of a
	// percent fails once the noise is this small, loose enough not to flake
	CHECK(delta < test::statistical_sigma_gate() * sigma);

	scene.destroy();
}

TEST_CASE("emitter nee scene: nee reduces variance at equal sample count") {
	// correctness alone is not the point of the feature. two independent
	// realizations of each config give a paired-noise estimate of each one's
	// own error, with no reference image needed.
	MorScene scene = build_floor_and_emitter(0.35f, 3.0f, 40.0f);
	REQUIRE(scene.emitterLightCount > 0u);

	RenderConfig cfg {};
	cfg.sampleCount = test::statistical_samples(512u);
	srat::CameraOrbit const cam = overhead_camera((f32)cfg.width, (f32)cfg.height);

	auto const noise = [&](bool const meshNee) {
		RenderConfig a = cfg;
		a.meshNee = meshNee;
		a.seedSalt = 1u;
		RenderConfig b = cfg;
		b.meshNee = meshNee;
		b.seedSalt = 2u;
		RenderResult const ra = render_scene(scene, cam, a);
		RenderResult const rb = render_scene(scene, cam, b);
		f64 sum = 0.0;
		for (size_t i = 0u; i < ra.rgb.size(); ++i) {
			f64 const d = (f64)ra.rgb[i] - (f64)rb.rgb[i];
			sum += d * d;
		}
		return std::sqrt(sum / (f64)ra.rgb.size());
	};

	f64 const noiseOn = noise(true);
	f64 const noiseOff = noise(false);
	CAPTURE(noiseOn);
	CAPTURE(noiseOff);
	// a small, bright emitter is exactly the case bsdf sampling handles worst
	CHECK(noiseOn < noiseOff);
	MESSAGE("nee noise ", noiseOn, " vs bsdf-only ", noiseOff);
}

TEST_CASE("emitter nee scene: zero emitters costs nothing and changes nothing") {
	// with no emitters the nee path must not consume rng draws, so the two
	// configs are bit-identical -- the cheapest possible proof that the
	// feature is inert when it has nothing to do
	std::vector<GpuMorMaterial> const materials {
		make_material({ 0.8f, 0.8f, 0.8f }, { 0.0f, 0.0f, 0.0f }, 0.0f),
	};
	std::vector<MeshTri> tris;
	push_quad(tris, 8.0f, 0.0f, 0u, false);
	MorScene scene = build_scene(tris, materials);
	CHECK(scene.emitterLightCount == 0u);

	RenderConfig cfg {};
	cfg.sampleCount = test::statistical_samples(64u);
	srat::CameraOrbit const cam = overhead_camera((f32)cfg.width, (f32)cfg.height);

	cfg.meshNee = true;
	RenderResult const on = render_scene(scene, cam, cfg);
	cfg.meshNee = false;
	RenderResult const off = render_scene(scene, cam, cfg);

	for (size_t i = 0u; i < on.rgb.size(); ++i) {
		REQUIRE(on.rgb[i] == off.rgb[i]);
	}

	scene.destroy();
}

TEST_CASE("emitter nee scene: bias localization sweep" * doctest::skip(false)) {
	// diagnostic, not a gate: measures the nee-on/off discrepancy in sigma
	// across the axes that separate the plausible causes.
	//   - propagationDepth 2 isolates direct lighting (one bounce). a bias
	//     present there is in the estimator at a single vertex; a bias that
	//     only appears at depth 8 is in the multi-bounce path state --
	//     previousVertexSampledLights, the delta flag, or the segment origin
	//     the backward pdf is measured from.
	//   - emitter size walks the hybrid sampling switch, since a large
	//     emitter takes the spherical branch and a small distant one the
	//     area branch.
	struct Case {
		char const * label;
		f32 halfSize;
		f32 height;
		f32 luminance;
		u32 depth;
	};
	Case const cases[] = {
		{ "depth 2, large emitter", 1.0f, 3.0f, 4.0f, 2u },
		{ "depth 3, large emitter", 1.0f, 3.0f, 4.0f, 3u },
		{ "depth 8, large emitter", 1.0f, 3.0f, 4.0f, 8u },
		{ "depth 2, small emitter", 0.15f, 3.0f, 200.0f, 2u },
		{ "depth 8, small emitter", 0.15f, 3.0f, 200.0f, 8u },
		{ "depth 2, emitter far (area branch)", 0.15f, 30.0f, 20000.0f, 2u },
	};

	for (Case const & c : cases) {
		MorScene scene = build_floor_and_emitter(c.halfSize, c.height, c.luminance);
		REQUIRE(scene.emitterLightCount > 0u);

		RenderConfig cfg {};
		cfg.sampleCount = test::statistical_samples(2048u);
		cfg.propagationDepth = c.depth;
		srat::CameraOrbit const cam = (
			overhead_camera((f32)cfg.width, (f32)cfg.height)
		);

		auto const run = [&](bool const meshNee, u32 const salt) {
			RenderConfig r = cfg;
			r.meshNee = meshNee;
			r.seedSalt = salt;
			return render_scene(scene, cam, r);
		};
		RenderResult const on1 = run(true, 11u);
		RenderResult const on2 = run(true, 29u);
		RenderResult const off1 = run(false, 11u);
		RenderResult const off2 = run(false, 29u);

		f64 const meanOn = 0.5 * (on1.mean + on2.mean);
		f64 const meanOff = 0.5 * (off1.mean + off2.mean);
		f64 const sigmaOn = 0.5 * std::abs(on1.mean - on2.mean);
		f64 const sigmaOff = 0.5 * std::abs(off1.mean - off2.mean);
		f64 const sigma = std::sqrt(sigmaOn * sigmaOn + sigmaOff * sigmaOff);
		f64 const delta = meanOn - meanOff;

		MESSAGE(
			std::string(c.label),
			": delta ", (meanOff > 0.0 ? delta / meanOff * 100.0 : 0.0), "%",
			"  (", (sigma > 0.0 ? std::abs(delta) / sigma : 0.0), " sigma)",
			"  on ", meanOn, " off ", meanOff
		);

		scene.destroy();
	}
}

// ---------------------------------------------------------------------------
// closed-form reference. the floor/emitter scene is single-bounce by
// construction -- a planar floor cannot see itself, and the emitter's base
// color is black so it reflects nothing -- so every floor pixel's value is
// exactly Lambert's analytic irradiance from the emitter quad, scaled by the
// diffuse brdf. that turns the on/off comparison from "these two disagree"
// into "this one is wrong", which is what an A/B alone can never say.
// ---------------------------------------------------------------------------

namespace {

// lambert's polygon irradiance: E = (L/2) * sum_edges theta_i * dot(n, e_i)
//
// computed entirely in double. the terms largely cancel -- the sum is the
// projected solid angle, which for a small emitter is orders of magnitude
// below the individual theta values -- and the corner vectors are nearly
// parallel there, so an f32 cross product loses most of its significant
// digits exactly where the reference is needed most
f64 polygon_irradiance_at(
	std::vector<f32v3> const & vertices, f32v3 const from, f32v3 const normal
) {
	struct D3 { f64 x, y, z; };
	auto const sub = [](f32v3 const & a, f32v3 const & b) {
		return D3 { (f64)a.x - (f64)b.x, (f64)a.y - (f64)b.y, (f64)a.z - (f64)b.z };
	};
	auto const norm = [](D3 const & v) {
		f64 const len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
		return len > 0.0 ? D3 { v.x / len, v.y / len, v.z / len } : D3 { 0.0, 0.0, 0.0 };
	};
	auto const dot = [](D3 const & a, D3 const & b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	};
	auto const cross = [](D3 const & a, D3 const & b) {
		return D3 {
			a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x,
		};
	};

	D3 const n { (f64)normal.x, (f64)normal.y, (f64)normal.z };
	f64 sum = 0.0;
	size_t const count = vertices.size();
	for (size_t i = 0u; i < count; ++i) {
		D3 const a = norm(sub(vertices[i], from));
		D3 const b = norm(sub(vertices[(i + 1u) % count], from));
		// 2*asin(|b-a|/2) rather than acos(dot): acos loses half its digits
		// as the angle goes to zero, which is the small-emitter case
		D3 const d { b.x - a.x, b.y - a.y, b.z - a.z };
		f64 const chord = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
		f64 const theta = 2.0 * std::asin(std::clamp(chord * 0.5, -1.0, 1.0));
		D3 const cr = cross(a, b);
		f64 const crLen = std::sqrt(cr.x * cr.x + cr.y * cr.y + cr.z * cr.z);
		if (crLen <= 0.0) { continue; }
		sum += theta * dot(n, D3 { cr.x / crLen, cr.y / crLen, cr.z / crLen });
	}
	// the formula's sign tracks the polygon's winding as seen from the
	// receiver; irradiance itself is unsigned, and emission here is two-sided
	return std::abs(0.5 * sum);
}

// general 4x4 inverse; srat only carries the analytic inverse of a composed
// TRS, and the shader inverts a full view-projection
f32m44 invert_m44(f32m44 const & m) {
	f32 const * const a = m.m.data;
	f32 inv[16];
	inv[0] = a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15]
		+ a[9]*a[7]*a[14] + a[13]*a[6]*a[11] - a[13]*a[7]*a[10];
	inv[4] = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15]
		- a[8]*a[7]*a[14] - a[12]*a[6]*a[11] + a[12]*a[7]*a[10];
	inv[8] = a[4]*a[9]*a[15] - a[4]*a[11]*a[13] - a[8]*a[5]*a[15]
		+ a[8]*a[7]*a[13] + a[12]*a[5]*a[11] - a[12]*a[7]*a[9];
	inv[12] = -a[4]*a[9]*a[14] + a[4]*a[10]*a[13] + a[8]*a[5]*a[14]
		- a[8]*a[6]*a[13] - a[12]*a[5]*a[10] + a[12]*a[6]*a[9];
	inv[1] = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15]
		- a[9]*a[3]*a[14] - a[13]*a[2]*a[11] + a[13]*a[3]*a[10];
	inv[5] = a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15]
		+ a[8]*a[3]*a[14] + a[12]*a[2]*a[11] - a[12]*a[3]*a[10];
	inv[9] = -a[0]*a[9]*a[15] + a[0]*a[11]*a[13] + a[8]*a[1]*a[15]
		- a[8]*a[3]*a[13] - a[12]*a[1]*a[11] + a[12]*a[3]*a[9];
	inv[13] = a[0]*a[9]*a[14] - a[0]*a[10]*a[13] - a[8]*a[1]*a[14]
		+ a[8]*a[2]*a[13] + a[12]*a[1]*a[10] - a[12]*a[2]*a[9];
	inv[2] = a[1]*a[6]*a[15] - a[1]*a[7]*a[14] - a[5]*a[2]*a[15]
		+ a[5]*a[3]*a[14] + a[13]*a[2]*a[7] - a[13]*a[3]*a[6];
	inv[6] = -a[0]*a[6]*a[15] + a[0]*a[7]*a[14] + a[4]*a[2]*a[15]
		- a[4]*a[3]*a[14] - a[12]*a[2]*a[7] + a[12]*a[3]*a[6];
	inv[10] = a[0]*a[5]*a[15] - a[0]*a[7]*a[13] - a[4]*a[1]*a[15]
		+ a[4]*a[3]*a[13] + a[12]*a[1]*a[7] - a[12]*a[3]*a[5];
	inv[14] = -a[0]*a[5]*a[14] + a[0]*a[6]*a[13] + a[4]*a[1]*a[14]
		- a[4]*a[2]*a[13] - a[12]*a[1]*a[6] + a[12]*a[2]*a[5];
	inv[3] = -a[1]*a[6]*a[11] + a[1]*a[7]*a[10] + a[5]*a[2]*a[11]
		- a[5]*a[3]*a[10] - a[9]*a[2]*a[7] + a[9]*a[3]*a[6];
	inv[7] = a[0]*a[6]*a[11] - a[0]*a[7]*a[10] - a[4]*a[2]*a[11]
		+ a[4]*a[3]*a[10] + a[8]*a[2]*a[7] - a[8]*a[3]*a[6];
	inv[11] = -a[0]*a[5]*a[11] + a[0]*a[7]*a[9] + a[4]*a[1]*a[11]
		- a[4]*a[3]*a[9] - a[8]*a[1]*a[7] + a[8]*a[3]*a[5];
	inv[15] = a[0]*a[5]*a[10] - a[0]*a[6]*a[9] - a[4]*a[1]*a[10]
		+ a[4]*a[2]*a[9] + a[8]*a[1]*a[6] - a[8]*a[2]*a[5];
	f32 det = a[0]*inv[0] + a[1]*inv[4] + a[2]*inv[8] + a[3]*inv[12];
	REQUIRE(std::abs(det) > 0.0f);
	det = 1.0f / det;
	f32m44 out {};
	for (i32 i = 0; i < 16; ++i) { out.m[i] = inv[i] * det; }
	return out;
}

// reconstructs the camera ray exactly as emitter_nee_scene.comp does, so a
// mismatch here would be a harness bug rather than a renderer one
f32v3 pixel_ray_direction(
	srat::CameraOrbit const & cam, u32 const x, u32 const y,
	u32 const width, u32 const height
) {
	f32m44 const viewProj = (
		srat::camera_orbit_proj(cam) * srat::camera_orbit_view(cam)
	);
	f32m44 const invVP = invert_m44(viewProj);
	f32 const u = ((f32)x + 0.5f) / (f32)width;
	f32 const v = ((f32)y + 0.5f) / (f32)height;
	f32v4 const ndc { u * 2.0f - 1.0f, v * 2.0f - 1.0f, 1.0f, 1.0f };
	f32v4 const worldFar = invVP * ndc;
	f32v3 const far {
		worldFar.x / worldFar.w, worldFar.y / worldFar.w, worldFar.z / worldFar.w
	};
	return f32v3_normalize(far - srat::camera_orbit_eye(cam));
}

} // namespace

TEST_CASE("emitter nee scene: matches lambert's closed form") {
	f32 const emitterHalfSize = 1.0f;
	f32 const emitterHeight = 3.0f;
	f32 const emissionLuminance = 4.0f;
	f32 const floorAlbedo = 0.8f;
	f32 const floorHalfSize = 8.0f;

	MorScene scene = (
		build_floor_and_emitter(
			emitterHalfSize, emitterHeight, emissionLuminance
		)
	);
	REQUIRE(scene.emitterLightCount > 0u);

	RenderConfig cfg {};
	cfg.sampleCount = test::statistical_samples(4096u);
	srat::CameraOrbit const cam = overhead_camera((f32)cfg.width, (f32)cfg.height);

	cfg.meshNee = true;
	RenderResult const on = render_scene(scene, cam, cfg);
	cfg.meshNee = false;
	RenderResult const off = render_scene(scene, cam, cfg);

	std::vector<f32v3> const quad {
		f32v3 { -emitterHalfSize, emitterHeight, -emitterHalfSize },
		f32v3 { emitterHalfSize, emitterHeight, -emitterHalfSize },
		f32v3 { emitterHalfSize, emitterHeight, emitterHalfSize },
		f32v3 { -emitterHalfSize, emitterHeight, emitterHalfSize },
	};
	f32v3 const eye = srat::camera_orbit_eye(cam);
	f32v3 const floorNormal { 0.0f, 1.0f, 0.0f };
	u32 const pixelCount = cfg.width * cfg.height;

	f64 sumAnalytic = 0.0;
	f64 sumOn = 0.0;
	f64 sumOff = 0.0;
	u32 compared = 0u;
	for (u32 y = 0u; y < cfg.height; ++y) {
		for (u32 x = 0u; x < cfg.width; ++x) {
			f32v3 const dir = pixel_ray_direction(cam, x, y, cfg.width, cfg.height);
			if (dir.y >= -1e-6f) { continue; }
			f32 const t = -eye.y / dir.y;
			if (t <= 0.0f) { continue; }
			f32v3 const p {
				eye.x + dir.x * t, 0.0f, eye.z + dir.z * t
			};
			// floor only, and skip pixels whose ray passes through the
			// emitter quad on the way down (those show emission, not floor)
			if (
				std::abs(p.x) > floorHalfSize || std::abs(p.z) > floorHalfSize
			) {
				continue;
			}
			f32 const tEmitter = (emitterHeight - eye.y) / dir.y;
			if (tEmitter > 0.0f && tEmitter < t) {
				f32 const ex = eye.x + dir.x * tEmitter;
				f32 const ez = eye.z + dir.z * tEmitter;
				if (
					std::abs(ex) <= emitterHalfSize
					&& std::abs(ez) <= emitterHalfSize
				) {
					continue;
				}
			}

			f64 const irradiance = polygon_irradiance_at(quad, p, floorNormal);
			f64 const analytic = (
				irradiance * (f64)emissionLuminance * (f64)floorAlbedo / M_PI
			);
			u32 const pixel = y * cfg.width + x;
			sumAnalytic += analytic;
			sumOn += (f64)on.rgb[pixel];
			sumOff += (f64)off.rgb[pixel];
			++compared;
		}
	}

	REQUIRE(compared > 500u);
	f64 const meanAnalytic = sumAnalytic / (f64)compared;
	f64 const meanOn = sumOn / (f64)compared;
	f64 const meanOff = sumOff / (f64)compared;

	MESSAGE(
		"floor pixels ", compared,
		"  analytic ", meanAnalytic,
		"  nee-on ", meanOn, " (", (meanOn / meanAnalytic - 1.0) * 100.0, "%)",
		"  nee-off ", meanOff, " (", (meanOff / meanAnalytic - 1.0) * 100.0, "%)"
	);

	// explicit relative error: doctest's Approx adds a scale of 1.0 to the
	// denominator, so epsilon(0.01) against a value of ~0.14 is really a ~7%
	// tolerance -- loose enough to pass a 2% bias unnoticed
	f64 const relOn = std::abs(meanOn - meanAnalytic) / meanAnalytic;
	f64 const relOff = std::abs(meanOff - meanAnalytic) / meanAnalytic;
	CAPTURE(relOn);
	CAPTURE(relOff);
	CHECK(relOn < 0.01);
	CHECK(relOff < 0.01);

	scene.destroy();
}

// ---------------------------------------------------------------------------
// the same scenes through the wavefront core (lib/ponder/shaders), driven by
// ponder::Wavefront exactly as the viewer drives it. the old-pipeline cases
// above test shaders-old; these are the regression suite for the new core
// (plan-many-lights.md section 2.4, phase P-1b)
// ---------------------------------------------------------------------------
namespace {

// mirrors ReadbackPc in wavefront-readback.slang
struct WavefrontReadbackPc {
	u64 sumVa;
	u32 imageHandle;
	u32 width;
	u32 height;
	u32 pad0;
};

struct WavefrontRenderConfig {
	u32 width = 64u;
	u32 height = 64u;
	u32 sampleCount = 512u;
	bool spectral = false;
	bool meshNee = true;
	u32 seedOffset = 0u;
	ponder::WavefrontEnvironmentMode environmentMode = (
		ponder::WavefrontEnvironmentMode::Black
	);
	f32 environmentIntensity = 0.0f;
	// records debugPixel's path on the last frame when set
	ponder::DebugPath * debugPathOut = nullptr;
	u32v2 debugPixel { 0u, 0u };
	bool regir = false;
	bool regirJitter = false;
};

struct WavefrontView {
	f32v3 position;
	f32v3 forward;
	f32 fovY;
};

WavefrontView wavefront_view_from_orbit(srat::CameraOrbit const & cam) {
	f32v3 const eye = srat::camera_orbit_eye(cam);
	return WavefrontView {
		.position = eye,
		.forward = f32v3_normalize(cam.target - eye),
		.fovY = cam.fovY,
	};
}

struct PixelXyz {
	f64 x;
	f64 y;
	f64 z;
};

// per-pixel mean of cie xyz over sampleCount frames
std::vector<PixelXyz> render_scene_wavefront_xyz(
	MorScene const & scene,
	WavefrontView const & view,
	WavefrontRenderConfig const & cfg
) {
	ponder::Wavefront wavefront = ponder::wavefront_create();
	ponder::Regir regir = ponder::regir_create();
	ponder::wavefront_resize(
		wavefront, cfg.width, cfg.height, cfg.width * cfg.height
	);
	if (cfg.debugPathOut != nullptr) {
		ponder::wavefront_debug_path_prepare(wavefront);
	}
	REQUIRE(wavefront.integrate.spectral.id != 0u);
	REQUIRE(wavefront.integrate.rgb.id != 0u);

	char const * const includePaths[] = { TEST_SHADER_DIR };
	vkof::Pipeline const readbackPipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = TEST_SHADER_DIR "slang/entry/wavefront-readback.slang",
			.includePaths = srat::slice<char const * const>(includePaths, 1u),
		})
	);
	REQUIRE(readbackPipeline.id != 0u);

	u32 const pixelCount = cfg.width * cfg.height;
	std::vector<f32v4> const zeros(pixelCount, f32v4 { 0.0f, 0.0f, 0.0f, 0.0f });
	vkof::Buffer sums = (
		test::upload_bytes(zeros.data(), zeros.size() * sizeof(f32v4))
	);

	vkof::TransientImage const sampleImage = (
		vkof::transient_image_create({
			.format = vkof::ImageFormat::r32g32b32a32_sfloat,
			.scaleWidth = 1.0f,
			.scaleHeight = 1.0f,
			.mipLevels = 1u,
			.isDoubleBuffered = false,
		})
	);

	vkof::acceleration_structure_set_tlas(scene.tlas);

	ponder::WavefrontCamera const camera {
		.position = view.position,
		.forward = view.forward,
		.fovY = view.fovY,
	};
	GpuGlobalPc const rootPc {
		.time = 0.0f,
		.cameraPos = view.position,
		.exposure = 1.0f,
		.pad0 = 0.0f,
		.viewProj = f32m44_identity(),
		.extended = 0u,
		.models = vkof::buffer_virtual_address(scene.models),
		.pad1 = wavefront.globalVa,
		.pad2 = 0u,
		.pad3 = 0u,
	};

	for (u32 s = 0u; s < cfg.sampleCount; ++s) {
		u32 const sampleIndex = cfg.seedOffset + s;
		bool const recordPath = (
			cfg.debugPathOut != nullptr && s + 1u == cfg.sampleCount
		);
		ponder::wavefront_global_update(
			wavefront,
			regir,
			ponder::WavefrontGlobalInfo {
				.camera = camera,
				.blackhole = ponder::WavefrontBlackhole {},
				.ambientMedium = ponder::WavefrontAmbientMedium {
					.type = ponder::WavefrontAmbientMediumType::None,
					.sigmaAbsorption = f32v3 { 0.0f, 0.0f, 0.0f },
					.sigmaScattering = f32v3 { 0.0f, 0.0f, 0.0f },
					.phase = ponder::WavefrontAmbientMediumPhase::HenyeyGreenstein,
					.anisotropy = 0.0f,
					.draineAlpha = 0.0f,
					.dropletDiameter = 0.0f,
					.heightOrigin = 0.0f,
					.heightFalloff = 1.0f,
				},
				.materialMediumTableVa = 0u,
				.materialMediumRecordsVa = 0u,
				.materialMediumModelCount = 0u,
				.sceneEmitterLightsVa = (
					scene.emitterLights.id != 0u
						? vkof::buffer_virtual_address(scene.emitterLights)
						: 0u
				),
				.sceneEmitterLightCount = scene.emitterLightCount,
				.meshNeeEnabled = cfg.meshNee,
				.envNeeEnabled = false,
				.environment = ponder::WavefrontEnvironment {
					.mode = cfg.environmentMode,
					.intensity = cfg.environmentIntensity,
					.rotation = 0.0f,
					.radianceHandle = 0u,
					.pdfHandle = 0u,
					.aliasProbHandle = 0u,
					.aliasIndexHandle = 0u,
					.emitterIndex = 0xFFFFFFFFu,
				},
				.groundHeating = ponder::WavefrontGroundHeating {},
				.regir = ponder::RegirSettings {
					.enabled = cfg.regir,
					.cellSize = 0.5f,
					.fillIndex = sampleIndex,
					.maxAge = 0u,
					.refresh = true,
					.jitter = cfg.regirJitter,
				},
				.modelsVa = vkof::buffer_virtual_address(scene.models),
				.shutterOpen = 0.0f,
				.shutterDuration = 0.0f,
				.frameIndex = sampleIndex,
				.sampleIndex = sampleIndex,
				.width = cfg.width,
				.height = cfg.height,
				.region = std::nullopt,
				.debugPath = recordPath,
				.debugPixel = cfg.debugPixel,
				.debugCamera = camera,
			}
		);
		u32 const sampleHandle = (
			vkof::transient_image_storage_handle({
				.image = sampleImage,
				.mipLevel = 0u,
			})
		);
		ponder::WavefrontIntegrateResult const integrate = (
			ponder::wavefront_integrate(
				wavefront,
				ponder::WavefrontIntegrateInfo {
					.outputImage = sampleImage,
					.outputHandle = sampleHandle,
					.aov = std::nullopt,
					.rrGuide = std::nullopt,
					.globalJitter = std::nullopt,
					.spectral = cfg.spectral,
					.debugPath = recordPath,
					.width = cfg.width,
					.height = cfg.height,
					.stepBudget = ponder::skWavefrontPassStepCount,
				}
			)
		);
		REQUIRE(integrate.passComplete);
		vkof::RenderNode const integrateNode = integrate.node;
		vkof::RenderNode const readbackNode = (
			vkof::render_node_create({ .queue = vkof::CommandQueue::graphics })
		);
		vkof::render_node_add_image({
			.node = readbackNode,
			.image = sampleImage,
			.access = vkof::RenderNodeAccess::read,
		});
		WavefrontReadbackPc const readbackPc {
			.sumVa = vkof::buffer_virtual_address(sums),
			.imageHandle = sampleHandle,
			.width = cfg.width,
			.height = cfg.height,
			.pad0 = 0u,
		};
		vkof::render_node_callback({
			.node = readbackNode,
			.callback = [=](vkof::CommandBuffer const & cmd) {
				vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
					.cmd = cmd,
					.pipeline = readbackPipeline,
					.push = readbackPc,
					.threadgroupSize = u32v3 { 8u, 8u, 1u },
					.invocationCount = u32v3 { cfg.width, cfg.height, 1u },
				});
			},
		});
		vkof::RenderNode nodes[3] {};
		u32 nodeCount = 0u;
		vkof::RenderNode regirFillNode { 0u };
		if (cfg.regir) {
			regirFillNode = ponder::regir_fill(regir);
			nodes[nodeCount++] = regirFillNode;
		}
		nodes[nodeCount++] = integrateNode;
		nodes[nodeCount++] = readbackNode;
		vkof::render_graph_execute({
			.nodes = srat::slice<vkof::RenderNode const>(nodes, nodeCount),
			.rootPushconstant = srat::slice_as_bytes(rootPc),
		});
		vkof::render_node_destroy(readbackNode);
		vkof::render_node_destroy(integrateNode);
		if (regirFillNode.id != 0u) {
			vkof::render_node_destroy(regirFillNode);
		}
		// the next global_update overwrites the buffer this frame reads
		test::gpu_wait();
		if (recordPath) {
			ponder::wavefront_debug_path_read(wavefront, *cfg.debugPathOut);
		}
	}

	std::vector<f32v4> const raw = test::readback<f32v4>(sums, 0u, pixelCount);
	std::vector<PixelXyz> xyz(pixelCount, PixelXyz { 0.0, 0.0, 0.0 });
	u32 countMismatch = 0u;
	for (u32 i = 0u; i < pixelCount; ++i) {
		f64 const n = (f64)cfg.sampleCount;
		xyz[i] = PixelXyz {
			.x = (f64)raw[i].x / n,
			.y = (f64)raw[i].y / n,
			.z = (f64)raw[i].z / n,
		};
		if (raw[i].w != (f32)cfg.sampleCount) {
			++countMismatch;
		}
	}
	// integrate writes w = 1 per frame; vkof hands back a live pipeline handle
	// even when a shader failed to compile, so this is the guard that either
	// shader ran
	REQUIRE_MESSAGE(
		countMismatch == 0u,
		"wavefront integrate or readback did not run; see the compile log above"
	);

	vkof::buffer_destroy(sums);
	vkof::pipeline_destroy(readbackPipeline);
	ponder::regir_destroy(regir);
	ponder::wavefront_destroy(wavefront);
	return xyz;
}

// per-pixel mean of Y (luminance) over sampleCount frames
std::vector<f64> render_scene_wavefront(
	MorScene const & scene,
	WavefrontView const & view,
	WavefrontRenderConfig const & cfg
) {
	std::vector<PixelXyz> const xyz = (
		render_scene_wavefront_xyz(scene, view, cfg)
	);
	std::vector<f64> luminance(xyz.size(), 0.0);
	for (u32 i = 0u; i < (u32)xyz.size(); ++i) {
		luminance[i] = xyz[i].y;
	}
	return luminance;
}

// Camera_Pinhole's ray for a sub-pixel position, with the basis
// wavefront_global_update builds (util/camera.slang, wavefront.cpp)
f32v3 wavefront_ray_direction(
	WavefrontView const & view,
	f32 const px,
	f32 const py,
	u32 const width,
	u32 const height
) {
	f32 const aspect = (f32)width / (f32)height;
	f32v3 const forward = f32v3_normalize(view.forward);
	f32v3 const right = (
		f32v3_normalize(f32v3_cross(forward, f32v3 { 0.0f, 1.0f, 0.0f }))
	);
	f32v3 const up = f32v3_cross(right, forward);
	f32 const tanHalfFovY = std::tan(view.fovY * 0.5f);
	f32 const ndcX = (px / (f32)width) * 2.0f - 1.0f;
	f32 const ndcY = 1.0f - (py / (f32)height) * 2.0f;
	return (
		f32v3_normalize(
			forward
			+ right * (ndcX * tanHalfFovY * aspect)
			+ up * (ndcY * tanHalfFovY)
		)
	);
}

} // namespace

TEST_CASE("wavefront nee scene: matches lambert's closed form") {
	f32 const emitterHalfSize = 1.0f;
	f32 const emitterHeight = 3.0f;
	f32 const emissionLuminance = 4.0f;
	f32 const floorAlbedo = 0.8f;
	f32 const floorHalfSize = 8.0f;
	// the integrator jitters inside the pixel, so the reference averages
	// the closed form over the same footprint
	u32 const skSubpixel = 4u;

	MorScene scene = (
		build_floor_and_emitter(
			emitterHalfSize, emitterHeight, emissionLuminance
		)
	);
	REQUIRE(scene.emitterLightCount > 0u);

	WavefrontRenderConfig cfg {};
	cfg.sampleCount = test::statistical_samples(4096u);
	WavefrontView const view = (
		wavefront_view_from_orbit(
			overhead_camera((f32)cfg.width, (f32)cfg.height)
		)
	);

	std::vector<f32v3> const quad {
		f32v3 { -emitterHalfSize, emitterHeight, -emitterHalfSize },
		f32v3 { emitterHalfSize, emitterHeight, -emitterHalfSize },
		f32v3 { emitterHalfSize, emitterHeight, emitterHalfSize },
		f32v3 { -emitterHalfSize, emitterHeight, emitterHalfSize },
	};
	f32v3 const floorNormal { 0.0f, 1.0f, 0.0f };
	f32v3 const eye = view.position;

	// -- analytic per pixel; NaN marks a pixel whose footprint is not
	// entirely lit floor (emitter silhouette, floor edge, sky)
	u32 const pixelCount = cfg.width * cfg.height;
	std::vector<f64> analytic(pixelCount, std::nan(""));
	for (u32 y = 0u; y < cfg.height; ++y) {
		for (u32 x = 0u; x < cfg.width; ++x) {
			f64 sum = 0.0;
			bool usable = true;
			for (u32 sy = 0u; sy < skSubpixel && usable; ++sy) {
				for (u32 sx = 0u; sx < skSubpixel && usable; ++sx) {
					f32 const px = (
						(f32)x + ((f32)sx + 0.5f) / (f32)skSubpixel
					);
					f32 const py = (
						(f32)y + ((f32)sy + 0.5f) / (f32)skSubpixel
					);
					f32v3 const dir = (
						wavefront_ray_direction(view, px, py, cfg.width, cfg.height)
					);
					if (dir.y >= -1e-6f) {
						usable = false;
						break;
					}
					f32 const t = -eye.y / dir.y;
					f32v3 const p { eye.x + dir.x * t, 0.0f, eye.z + dir.z * t };
					if (
						std::abs(p.x) > floorHalfSize
						|| std::abs(p.z) > floorHalfSize
					) {
						usable = false;
						break;
					}
					f32 const tEmitter = (emitterHeight - eye.y) / dir.y;
					if (tEmitter > 0.0f && tEmitter < t) {
						f32 const ex = eye.x + dir.x * tEmitter;
						f32 const ez = eye.z + dir.z * tEmitter;
						if (
							std::abs(ex) <= emitterHalfSize
							&& std::abs(ez) <= emitterHalfSize
						) {
							usable = false;
							break;
						}
					}
					sum += (
						polygon_irradiance_at(quad, p, floorNormal)
						* (f64)emissionLuminance * (f64)floorAlbedo / M_PI
					);
				}
			}
			if (usable) {
				analytic[y * cfg.width + x] = (
					sum / (f64)(skSubpixel * skSubpixel)
				);
			}
		}
	}

	struct Variant {
		char const * name;
		bool spectral;
		bool meshNee;
		bool regir;
	};
	Variant const variants[] = {
		{ "rgb nee-on", false, true, false },
		{ "rgb nee-off", false, false, false },
		{ "spectral nee-on", true, true, false },
		{ "spectral nee-off", true, false, false },
		{ "rgb regir", false, true, true },
		{ "spectral regir", true, true, true },
	};
	for (Variant const & variant : variants) {
		WavefrontRenderConfig variantCfg = cfg;
		variantCfg.spectral = variant.spectral;
		variantCfg.meshNee = variant.meshNee;
		variantCfg.regir = variant.regir;
		std::vector<f64> const rendered = (
			render_scene_wavefront(scene, view, variantCfg)
		);
		f64 sumAnalytic = 0.0;
		f64 sumRendered = 0.0;
		u32 compared = 0u;
		u32 nonFinite = 0u;
		for (u32 i = 0u; i < pixelCount; ++i) {
			if (!std::isfinite(rendered[i])) {
				++nonFinite;
				continue;
			}
			if (std::isnan(analytic[i])) {
				continue;
			}
			sumAnalytic += analytic[i];
			sumRendered += rendered[i];
			++compared;
		}
		REQUIRE(compared > 500u);
		f64 const meanAnalytic = sumAnalytic / (f64)compared;
		f64 const meanRendered = sumRendered / (f64)compared;
		f64 const rel = std::abs(meanRendered - meanAnalytic) / meanAnalytic;
		std::string const name = variant.name;
		MESSAGE(
			name,
			"  floor pixels ", compared,
			"  analytic ", meanAnalytic,
			"  rendered ", meanRendered,
			" (", (meanRendered / meanAnalytic - 1.0) * 100.0, "%)"
		);
		CAPTURE(name);
		CAPTURE(rel);
		CHECK(nonFinite == 0u);
		CHECK(rel < 0.01);
	}

	scene.destroy();
}

TEST_CASE("wavefront nee scene: regir matches the power cdf under an occluder") {
	f32 const emitterHalfSize = 1.0f;
	f32 const emitterHeight = 3.0f;
	f32 const occluderHeight = 1.5f;
	std::vector<GpuMorMaterial> const materials {
		make_material({ 0.8f, 0.8f, 0.8f }, { 0.0f, 0.0f, 0.0f }, 0.0f),
		make_material({ 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 4.0f),
		make_material({ 0.5f, 0.5f, 0.5f }, { 0.0f, 0.0f, 0.0f }, 0.0f),
	};
	std::vector<MeshTri> tris;
	push_quad(tris, 8.0f, 0.0f, 0u, false);
	push_quad(tris, emitterHalfSize, emitterHeight, 1u, true);
	f32v3 const o00 { -4.0f, occluderHeight, -4.0f };
	f32v3 const o10 { 0.0f, occluderHeight, -4.0f };
	f32v3 const o11 { 0.0f, occluderHeight, 4.0f };
	f32v3 const o01 { -4.0f, occluderHeight, 4.0f };
	tris.emplace_back(MeshTri { o00, o11, o10, 2u });
	tris.emplace_back(MeshTri { o00, o01, o11, 2u });
	MorScene scene = build_scene(tris, materials);
	REQUIRE(scene.emitterLightCount > 0u);

	WavefrontRenderConfig cfg {};
	cfg.sampleCount = test::statistical_samples(4096u);
	WavefrontView const view = (
		wavefront_view_from_orbit(
			overhead_camera((f32)cfg.width, (f32)cfg.height)
		)
	);
	auto const fnMean = [&](bool const regir, bool const jitter) -> f64 {
		WavefrontRenderConfig variantCfg = cfg;
		variantCfg.regir = regir;
		variantCfg.regirJitter = jitter;
		std::vector<f64> const rendered = (
			render_scene_wavefront(scene, view, variantCfg)
		);
		f64 sum = 0.0;
		for (f64 const v : rendered) {
			REQUIRE(std::isfinite(v));
			sum += v;
		}
		return sum / (f64)rendered.size();
	};
	f64 const meanCdf = fnMean(false, false);
	f64 const meanRegir = fnMean(true, false);
	f64 const rel = std::abs(meanRegir - meanCdf) / meanCdf;
	MESSAGE(
		"power cdf ", meanCdf,
		"  regir ", meanRegir,
		" (", (meanRegir / meanCdf - 1.0) * 100.0, "%)"
	);
	CAPTURE(rel);
	CHECK(rel < 0.01);

	// -- jittered cell lookup borrows neighbouring cells across the occluder
	// edge; still unbiased
	f64 const meanJitter = fnMean(true, true);
	f64 const relJitter = std::abs(meanJitter - meanCdf) / meanCdf;
	MESSAGE(
		"regir jitter ", meanJitter,
		" (", (meanJitter / meanCdf - 1.0) * 100.0, "%)"
	);
	CAPTURE(relJitter);
	CHECK(relJitter < 0.01);

	// -- nee must actually have taken the regir branch at the primary hit
	ponder::DebugPath path {};
	WavefrontRenderConfig pathCfg = cfg;
	pathCfg.regir = true;
	pathCfg.sampleCount = 8u;
	pathCfg.debugPathOut = &path;
	pathCfg.debugPixel = u32v2 { 32u, 32u };
	(void)render_scene_wavefront(scene, view, pathCfg);
	REQUIRE(path.vertexCount > 0u);
	ponder::DebugPathVertex const & primary = path.vertices[0];
	CAPTURE(primary.neeStrategy);
	CAPTURE(primary.regirSlot);
	CHECK(primary.regirSlot != ponder::skWavefrontRegirSlotInvalid);
	scene.destroy();
}

TEST_CASE("wavefront nee scene: debug path records the nee connection") {
	MorScene scene = build_floor_and_emitter(1.0f, 3.0f, 4.0f);
	REQUIRE(scene.emitterLightCount > 0u);
	WavefrontRenderConfig cfg {};
	WavefrontView const view = (
		wavefront_view_from_orbit(
			overhead_camera((f32)cfg.width, (f32)cfg.height)
		)
	);
	u32 recordedVertices = 0u;
	u32 neeAttempts = 0u;
	u32 neeVisible = 0u;
	for (u32 s = 0u; s < 8u; ++s) {
		ponder::DebugPath path {};
		WavefrontRenderConfig pathCfg = cfg;
		pathCfg.sampleCount = 1u;
		pathCfg.seedOffset = s;
		pathCfg.debugPathOut = &path;
		pathCfg.debugPixel = u32v2 { 32u, 32u };
		(void)render_scene_wavefront(scene, view, pathCfg);
		REQUIRE(path.byteCount == (u32)sizeof(ponder::DebugPath));
		u32 const vertexCount = (
			std::min(path.vertexCount, ponder::skDebugPathVertexCapacity)
		);
		recordedVertices += vertexCount;
		for (u32 v = 0u; v < vertexCount; ++v) {
			ponder::DebugPathVertex const & d = path.vertices[v];
			if (d.event != 0u || d.bounce != 0u) {
				continue;
			}
			CAPTURE(s);
			CAPTURE(d.neeState);
			CAPTURE(d.neeStrategy);
			CHECK(d.neeStrategy == ponder::skDebugPathNeeStrategyPowerCdf);
			CHECK(d.regirSlot == ponder::skWavefrontRegirSlotInvalid);
			if (d.neeState != ponder::skDebugPathNeeNone) {
				++neeAttempts;
			}
			if (d.neeState == ponder::skDebugPathNeeVisible) {
				++neeVisible;
				f32v3 const toLight = d.neePosition - d.position;
				CHECK(f32v3_dot(toLight, toLight) > 0.0f);
			}
		}
	}
	MESSAGE(
		"vertices ", recordedVertices,
		"  nee attempts ", neeAttempts,
		"  visible ", neeVisible
	);
	REQUIRE(recordedVertices > 0u);
	CHECK(neeAttempts == 8u);
	CHECK(neeVisible == 8u);
	scene.destroy();
}

TEST_CASE("emitter nee scene: closed-form deficit vs incidence angle") {
	// diagnostic. the bsdf-only render sits ~0.75% under the closed form even
	// with nee off, so something in the production tracer loses energy
	// independently of this feature. tier 2 pinned openpbr's diffuse lobe to
	// lambert at 0.017% -- but at a single incidence angle. if the deficit
	// grows as the geometry moves off normal incidence, the lobe (or the
	// analytic model of it) is angle dependent; if it is flat, the loss is
	// elsewhere in the walk.
	struct Case {
		char const * label;
		f32 elevation;
		f32 emitterHeight;
	};
	Case const cases[] = {
		{ "near-normal (elevation 1.45)", 1.45f, 3.0f },
		{ "moderate (elevation 0.9)", 0.9f, 3.0f },
		{ "oblique (elevation 0.55)", 0.55f, 3.0f },
		{ "grazing (elevation 0.25)", 0.25f, 3.0f },
	};

	f32 const emitterHalfSize = 1.0f;
	f32 const emissionLuminance = 4.0f;
	f32 const floorAlbedo = 0.8f;
	f32 const floorHalfSize = 8.0f;

	for (Case const & c : cases) {
		MorScene scene = (
			build_floor_and_emitter(
				emitterHalfSize, c.emitterHeight, emissionLuminance
			)
		);
		RenderConfig cfg {};
		cfg.sampleCount = test::statistical_samples(4096u);
		cfg.meshNee = false;
		srat::CameraOrbit cam = (
			overhead_camera((f32)cfg.width, (f32)cfg.height)
		);
		cam.elevation = c.elevation;

		RenderResult const off = render_scene(scene, cam, cfg);
		cfg.meshNee = true;
		RenderResult const on = render_scene(scene, cam, cfg);

		std::vector<f32v3> const quad {
			f32v3 { -emitterHalfSize, c.emitterHeight, -emitterHalfSize },
			f32v3 { emitterHalfSize, c.emitterHeight, -emitterHalfSize },
			f32v3 { emitterHalfSize, c.emitterHeight, emitterHalfSize },
			f32v3 { -emitterHalfSize, c.emitterHeight, emitterHalfSize },
		};
		f32v3 const eye = srat::camera_orbit_eye(cam);
		f32v3 const floorNormal { 0.0f, 1.0f, 0.0f };

		f64 sumAnalytic = 0.0;
		f64 sumOn = 0.0;
		f64 sumOff = 0.0;
		u32 compared = 0u;
		for (u32 y = 0u; y < cfg.height; ++y) {
			for (u32 x = 0u; x < cfg.width; ++x) {
				f32v3 const dir = (
					pixel_ray_direction(cam, x, y, cfg.width, cfg.height)
				);
				if (dir.y >= -1e-6f) { continue; }
				f32 const t = -eye.y / dir.y;
				if (t <= 0.0f) { continue; }
				f32v3 const p { eye.x + dir.x * t, 0.0f, eye.z + dir.z * t };
				if (
					std::abs(p.x) > floorHalfSize || std::abs(p.z) > floorHalfSize
				) { continue; }
				f32 const tE = (c.emitterHeight - eye.y) / dir.y;
				if (tE > 0.0f && tE < t) {
					f32 const ex = eye.x + dir.x * tE;
					f32 const ez = eye.z + dir.z * tE;
					if (
						std::abs(ex) <= emitterHalfSize
						&& std::abs(ez) <= emitterHalfSize
					) { continue; }
				}
				u32 const pixel = y * cfg.width + x;
				sumAnalytic += (
					polygon_irradiance_at(quad, p, floorNormal)
					* (f64)emissionLuminance * (f64)floorAlbedo / M_PI
				);
				sumOn += (f64)on.rgb[pixel];
				sumOff += (f64)off.rgb[pixel];
				++compared;
			}
		}
		if (compared < 100u) { scene.destroy(); continue; }
		f64 const a = sumAnalytic / (f64)compared;
		MESSAGE(
			std::string(c.label), "  pixels ", compared,
			"  off ", (sumOff / (f64)compared / a - 1.0) * 100.0, "%",
			"  on ", (sumOn / (f64)compared / a - 1.0) * 100.0, "%"
		);
		scene.destroy();
	}
}

TEST_CASE("emitter nee scene: closed-form error vs emitter solid angle") {
	// the clean single-variable sweep. the camera and the floor region stay
	// fixed, so the only thing changing is the emitter's subtended solid
	// angle -- which is what drives the hybrid sampling switch and the
	// relative size of the light pdf against the bsdf pdf. emitted radiance
	// is scaled by 1/area so the analytic answer stays the same order across
	// the sweep and the percentages are comparable.
	f32 const emitterHeight = 3.0f;
	f32 const floorAlbedo = 0.8f;
	f32 const floorHalfSize = 8.0f;

	// at height 3 the spherical/area switch (3e-4 sr) sits near halfSize
	// 0.026, so the last entry is the only one in the area branch
	f32 const halfSizes[] = { 2.0f, 1.0f, 0.5f, 0.2f, 0.05f, 0.015f };

	srat::CameraOrbit const cam = overhead_camera(64.0f, 64.0f);
	f32v3 const eye = srat::camera_orbit_eye(cam);
	f32v3 const floorNormal { 0.0f, 1.0f, 0.0f };

	for (f32 const halfSize : halfSizes) {
		// keep total emitted power roughly constant across the sweep
		f32 const luminance = 4.0f / (halfSize * halfSize);
		MorScene scene = (
			build_floor_and_emitter(halfSize, emitterHeight, luminance)
		);
		REQUIRE(scene.emitterLightCount > 0u);

		RenderConfig cfg {};
		cfg.sampleCount = test::statistical_samples(4096u);
		cfg.meshNee = true;
		RenderResult const on = render_scene(scene, cam, cfg);
		cfg.meshNee = false;
		RenderResult const off = render_scene(scene, cam, cfg);

		std::vector<f32v3> const quad {
			f32v3 { -halfSize, emitterHeight, -halfSize },
			f32v3 { halfSize, emitterHeight, -halfSize },
			f32v3 { halfSize, emitterHeight, halfSize },
			f32v3 { -halfSize, emitterHeight, halfSize },
		};

		f64 sumAnalytic = 0.0;
		f64 sumOn = 0.0;
		f64 sumOff = 0.0;
		u32 compared = 0u;
		for (u32 y = 0u; y < cfg.height; ++y) {
			for (u32 x = 0u; x < cfg.width; ++x) {
				f32v3 const dir = (
					pixel_ray_direction(cam, x, y, cfg.width, cfg.height)
				);
				if (dir.y >= -1e-6f) { continue; }
				f32 const t = -eye.y / dir.y;
				if (t <= 0.0f) { continue; }
				f32v3 const p { eye.x + dir.x * t, 0.0f, eye.z + dir.z * t };
				if (
					std::abs(p.x) > floorHalfSize || std::abs(p.z) > floorHalfSize
				) { continue; }
				f32 const tE = (emitterHeight - eye.y) / dir.y;
				if (tE > 0.0f && tE < t) {
					f32 const ex = eye.x + dir.x * tE;
					f32 const ez = eye.z + dir.z * tE;
					if (std::abs(ex) <= halfSize && std::abs(ez) <= halfSize) {
						continue;
					}
				}
				u32 const pixel = y * cfg.width + x;
				sumAnalytic += (
					polygon_irradiance_at(quad, p, floorNormal)
					* (f64)luminance * (f64)floorAlbedo / M_PI
				);
				sumOn += (f64)on.rgb[pixel];
				sumOff += (f64)off.rgb[pixel];
				++compared;
			}
		}
		REQUIRE(compared > 100u);
		f64 const a = sumAnalytic / (f64)compared;
		// solid angle of the quad from directly beneath its centre, as a
		// rough label for which sampling branch dominates
		f64 const omega = (
			polygon_irradiance_at(
				quad, f32v3 { 0.0f, 0.0f, 0.0f }, floorNormal
			)
		);
		MESSAGE(
			"halfSize ", halfSize, " (proj.sr ~", omega, ")",
			"  off ", (sumOff / (f64)compared / a - 1.0) * 100.0, "%",
			"  on ", (sumOn / (f64)compared / a - 1.0) * 100.0, "%"
		);
		scene.destroy();
	}
}

TEST_CASE("emitter nee scene: nee connection accounting") {
	// instrumented run. nothing in these scenes stands between the floor and
	// the emitter, so the shadow ray should reject essentially no connection
	// that got as far as being traced. swept across emitter size because the
	// hybrid sampling branch changes with it, and a rejection rate that only
	// appears in one branch localises the fault to that branch.
	struct Case {
		char const * label;
		f32 halfSize;
	};
	Case const cases[] = {
		{ "large (spherical branch)", 1.0f },
		{ "small (spherical branch)", 0.05f },
		{ "tiny (area branch)", 0.015f },
	};

	for (Case const & c : cases) {
		MorScene scene = (
			build_floor_and_emitter(c.halfSize, 3.0f, 4.0f / (c.halfSize * c.halfSize))
		);
		REQUIRE(scene.emitterLightCount > 0u);

		RenderConfig cfg {};
		cfg.sampleCount = test::statistical_samples(256u);
		cfg.meshNee = true;
		cfg.collectNeeCounters = true;
		srat::CameraOrbit const cam = (
			overhead_camera((f32)cfg.width, (f32)cfg.height)
		);

		RenderResult const r = render_scene(scene, cam, cfg);
		REQUIRE(r.neeCounters.size() == 7u);

		f64 const attempts = (f64)r.neeCounters[0];
		REQUIRE(attempts > 0.0);
		f64 const valid = (f64)r.neeCounters[1];
		f64 const passedGate = (f64)r.neeCounters[2];
		f64 const occluded = (f64)r.neeCounters[3];
		f64 const contributed = (f64)r.neeCounters[4];

		MESSAGE(
			std::string(c.label),
			": attempts ", (u64)attempts,
			"  sampler-valid ", valid / attempts * 100.0, "%",
			"  passed f/cos ", passedGate / attempts * 100.0, "%",
			"  occluded ", occluded / std::max(passedGate, 1.0) * 100.0, "%",
			"  contributed ", contributed / attempts * 100.0, "%",
			"  | emissive-blocker ", (f64)r.neeCounters[5],
			" other-blocker ", (f64)r.neeCounters[6]
		);

		// an occluder-free scene: every traced connection should survive
		CHECK(occluded / std::max(passedGate, 1.0) < 0.001);
		scene.destroy();
	}
}

// ---------------------------------------------------------------------------
// tier 2.4 / 2.5 of plan-nee.md: the environment map folded into the same
// power-weighted light list as the emissive meshes. one pick, one shadow ray,
// and therefore one pick probability that has to appear in every env-side mis
// weight and divide back out of every env-side estimate. Getting that factor
// wrong is a silent, uniform darkening -- no NaN, no firefly, no failing
// mesh-only test -- which is why both halves of it get their own case.
// ---------------------------------------------------------------------------

TEST_CASE("emitter nee scene: env and mesh emitters combine linearly") {
	// transport is linear in the sources, so with identical geometry and
	// materials the combined render must equal the sum of the two
	// single-source renders. the emissive quad is black outside its emission,
	// so it is exactly the same occluder in all three configurations, and the
	// only thing that changes is which lights exist.
	EnvScene env = make_env_sky(ENVIRONMENT_OUTPUT_DIR "emitter_nee_sky.exr");

	MorScene lit = build_floor_and_emitter(1.0f, 3.0f, 4.0f);
	MorScene dark = build_floor_and_emitter(1.0f, 3.0f, 0.0f);
	REQUIRE(lit.emitterLightCount > 0u);
	REQUIRE(dark.emitterLightCount == 0u);

	f32 const envPower = env.tables.luminanceIntegral;
	REQUIRE(envPower > 0.0f);
	CombinedLightList meshOnlyList = make_light_list(lit.lightsCpu, 0.0f);
	CombinedLightList envOnlyList = make_light_list({}, envPower);
	CombinedLightList bothList = make_light_list(lit.lightsCpu, envPower);
	// the mesh-only list must not have grown an env entry, and the combined
	// one must have exactly one more entry than the mesh-only one
	REQUIRE(meshOnlyList.envLightIndex >= meshOnlyList.count);
	REQUIRE(envOnlyList.envLightIndex < envOnlyList.count);
	REQUIRE(bothList.count == meshOnlyList.count + 1u);

	RenderConfig base {};
	base.sampleCount = test::statistical_samples(2048u);
	base.seedSalt = 11u;
	srat::CameraOrbit const cam = (
		overhead_camera((f32)base.width, (f32)base.height)
	);

	auto const measure = [&](
		MorScene const & scene, bool const withEnv, CombinedLightList const & list
	) {
		RenderConfig cfg = base;
		cfg.envMapHandlesVa = withEnv ? env.va() : 0u;
		cfg.lightsOverride = list.buffer;
		cfg.lightsOverrideCount = list.count;
		cfg.envLightIndex = list.envLightIndex;
		return render_mean(scene, cam, cfg, /*realizations=*/4u);
	};

	MeanAndError const mesh = measure(lit, false, meshOnlyList);
	MeanAndError const envOnly = measure(dark, true, envOnlyList);
	MeanAndError const both = measure(lit, true, bothList);

	CHECK(mesh.nonFiniteCount == 0u);
	CHECK(envOnly.nonFiniteCount == 0u);
	CHECK(both.nonFiniteCount == 0u);
	REQUIRE(mesh.mean > 0.0);
	REQUIRE(envOnly.mean > 0.0);

	f64 const sum = mesh.mean + envOnly.mean;
	f64 const sigma = (
		std::sqrt(
			mesh.error * mesh.error
			+ envOnly.error * envOnly.error
			+ both.error * both.error
		)
	);
	f64 const delta = std::abs(both.mean - sum);

	MESSAGE(
		"combined ", both.mean, " vs mesh ", mesh.mean, " + env ", envOnly.mean,
		" = ", sum,
		"  delta ", delta / sum * 100.0, "%",
		"  sigma ", sigma / sum * 100.0, "%",
		"  = ", delta / std::max(sigma, 1e-12), " sigma"
	);
	CAPTURE(both.mean);
	CAPTURE(mesh.mean);
	CAPTURE(envOnly.mean);
	CAPTURE(sigma);
	CHECK(delta < test::statistical_sigma_gate() * sigma);

	bothList.destroy();
	envOnlyList.destroy();
	meshOnlyList.destroy();
	dark.destroy();
	lit.destroy();
	env.destroy();
}

// shared body of the two power-split sweeps below: three configurations that
// differ only in how the pick mass is divided, whose means must all agree
void check_split_sweep_invariant(
	MorScene const & scene,
	srat::CameraOrbit const & cam,
	RenderConfig const & base,
	std::string const & label,
	// builds the uploaded light list for a given share of the pick mass
	std::vector<vkof::Buffer> & ownedBuffers,
	std::vector<std::pair<u32, u32>> const & listCountAndEnvIndex,
	std::vector<f64> const & shares
) {
	std::vector<f64> means;
	std::vector<f64> errors;
	for (usize i = 0u; i < shares.size(); ++i) {
		RenderConfig cfg = base;
		cfg.lightsOverride = ownedBuffers[i];
		cfg.lightsOverrideCount = listCountAndEnvIndex[i].first;
		cfg.envLightIndex = listCountAndEnvIndex[i].second;
		MeanAndError const m = render_mean(scene, cam, cfg, /*realizations=*/4u);
		CHECK(m.nonFiniteCount == 0u);
		means.emplace_back(m.mean);
		errors.emplace_back(m.error);
		MESSAGE(
			label, " share ", shares[i], ": mean ", m.mean,
			"  sigma ", m.error / std::max(m.mean, 1e-12) * 100.0, "%"
		);
	}
	// every pair must agree: the extreme pair is what a bias proportional to
	// the share separates furthest, the middle one catches a non-monotonic
	// error that the extremes alone would miss
	for (usize i = 0u; i < means.size(); ++i) {
		for (usize j = i + 1u; j < means.size(); ++j) {
			f64 const sigma = (
				std::sqrt(errors[i] * errors[i] + errors[j] * errors[j])
			);
			f64 const delta = std::abs(means[i] - means[j]);
			CAPTURE(i);
			CAPTURE(j);
			CAPTURE(delta);
			CAPTURE(sigma);
			CHECK(delta < test::statistical_sigma_gate() * sigma);
		}
	}
}

TEST_CASE("emitter nee scene: the env/mesh power split does not move the mean") {
	// power steers the pick and nothing else, so forcing the environment's
	// share of the list to 0.1 / 0.5 / 0.9 must leave the rendered mean where
	// it was. a pick probability that is not divided back out, or one that is
	// missing from the backward mis weight, biases in proportion to the share
	// -- so it cannot survive a sweep that moves the share 9x in both
	// directions even though it would pass any single-split test.
	EnvScene env = make_env_sky(ENVIRONMENT_OUTPUT_DIR "emitter_nee_sky.exr");
	MorScene scene = build_floor_and_emitter(1.0f, 3.0f, 4.0f);
	REQUIRE(scene.emitterLightCount > 0u);

	f64 const meshPower = mesh_power_total(scene.lightsCpu);
	REQUIRE(meshPower > 0.0);

	RenderConfig base {};
	base.sampleCount = test::statistical_samples(2048u);
	base.seedSalt = 3u;
	base.envMapHandlesVa = env.va();
	srat::CameraOrbit const cam = (
		overhead_camera((f32)base.width, (f32)base.height)
	);

	std::vector<f64> const shares { 0.1, 0.5, 0.9 };
	std::vector<CombinedLightList> lists;
	std::vector<vkof::Buffer> buffers;
	std::vector<std::pair<u32, u32>> meta;
	for (f64 const share : shares) {
		// p_env = P_env / (P_mesh + P_env)
		CombinedLightList list = (
			make_light_list(
				scene.lightsCpu, (f32)(meshPower * share / (1.0 - share))
			)
		);
		REQUIRE(list.envLightIndex < list.count);
		buffers.emplace_back(list.buffer);
		meta.emplace_back(std::pair<u32, u32> { list.count, list.envLightIndex });
		lists.emplace_back(list);
	}

	check_split_sweep_invariant(
		scene, cam, base, "env", buffers, meta, shares
	);

	for (CombinedLightList & list : lists) { list.destroy(); }
	scene.destroy();
	env.destroy();
}

TEST_CASE("emitter nee scene: the split between two mesh emitters is inert") {
	// the same sweep with the environment taken out of it. Both entries are
	// ordinary mesh emitters sampled by the same triangle sampler, so the only
	// thing varying is the pick probability itself -- the env map's own
	// sampler, and the two defects test-env-mis.cpp documents in it, cannot
	// contribute. This is what attributes any trend in the case above.
	//
	// the powers are overwritten by hand rather than steered through geometry:
	// power is a pick weight and nothing else, so an entry whose power does
	// not match its true contribution is not an invalid list, only a badly
	// steered one -- exactly the property under test.
	std::vector<GpuMorMaterial> const materials {
		make_material({ 0.8f, 0.8f, 0.8f }, { 0.0f, 0.0f, 0.0f }, 0.0f),
		make_material({ 0.0f, 0.0f, 0.0f }, { 1.0f, 0.9f, 0.8f }, 4.0f),
		make_material({ 0.0f, 0.0f, 0.0f }, { 0.8f, 0.9f, 1.0f }, 4.0f),
	};
	// two quads side by side above the floor, each its own meshlet (and so its
	// own list entry) because the material run changes between them
	auto const pushQuadAt = [](
		std::vector<MeshTri> & out, f32 const cx, f32 const halfSize,
		f32 const height, u32 const materialIndex
	) {
		f32v3 const v00 { cx - halfSize, height, -halfSize };
		f32v3 const v10 { cx + halfSize, height, -halfSize };
		f32v3 const v11 { cx + halfSize, height, halfSize };
		f32v3 const v01 { cx - halfSize, height, halfSize };
		out.emplace_back(MeshTri { v00, v11, v10, materialIndex });
		out.emplace_back(MeshTri { v00, v01, v11, materialIndex });
	};
	std::vector<MeshTri> tris;
	push_quad(tris, 8.0f, 0.0f, /*materialIndex=*/0u, /*faceDown=*/false);
	pushQuadAt(tris, -1.5f, 0.8f, 3.0f, 1u);
	pushQuadAt(tris, 1.5f, 0.8f, 3.0f, 2u);
	MorScene scene = build_scene(tris, materials);
	REQUIRE(scene.emitterLightCount == 2u);

	RenderConfig base {};
	base.sampleCount = test::statistical_samples(2048u);
	base.seedSalt = 3u;
	srat::CameraOrbit const cam = (
		overhead_camera((f32)base.width, (f32)base.height)
	);

	std::vector<f64> const shares { 0.1, 0.5, 0.9 };
	std::vector<vkof::Buffer> buffers;
	std::vector<std::pair<u32, u32>> meta;
	for (f64 const share : shares) {
		std::vector<GpuEmitterLight> lights = scene.lightsCpu;
		lights[0].power = (f32)share;
		lights[1].power = (f32)(1.0 - share);
		(void)ponder::emitter_lights_finalize(lights);
		buffers.emplace_back(
			test::upload_bytes(
				lights.data(), lights.size() * sizeof(GpuEmitterLight)
			)
		);
		meta.emplace_back(
			std::pair<u32, u32> { (u32)lights.size(), 0xFFFFFFFFu }
		);
	}

	check_split_sweep_invariant(
		scene, cam, base, "first-emitter", buffers, meta, shares
	);

	for (vkof::Buffer & b : buffers) { vkof::buffer_destroy(b); }
	scene.destroy();
}

// ---------------------------------------------------------------------------
// tier 3.4 / 3.5 of plan-nee.md -- the two nee sites P4 added. Neither has
// prior art in cull, so neither inherits a working reference; the assertion is
// the same one the surface site is held to, applied one vertex deeper.
//
// Both are on/off mean-equality tests rather than closed-form ones. A random
// walk through a scattering volume has no analytic irradiance to check
// against, so the reference has to be the same scene integrated by the other
// technique -- which is exactly the property MIS unbiasedness claims.
// ---------------------------------------------------------------------------

TEST_CASE("emitter nee scene: on and off converge through a subsurface exit vertex") {
	// tier 3.4. A subsurface box under an emissive quad: every camera ray that
	// lands on the box enters the walk and leaves through an exit vertex, so
	// the exit-vertex nee block is on the critical path for most of the frame
	// rather than a rare corner.
	//
	// The exit's own mis pairs openPbrTransmissionEvaluateF/Pdf against the
	// emitter density, and the *next* vertex balances against the state the
	// walk records on the way out. Both halves have to price the same
	// connection identically or the box reads brighter or darker than the
	// bsdf-only integration of the identical scene.
	//
	// Swept over the exit interface's roughness because that is the axis the
	// two defects found here scaled with: a matched ior (specularWeight 0,
	// which drives openPbrEffectiveIor to exactly 1) made the rough btdf
	// degenerate and read +73%, and gating nee on the sampled microfacet's tir
	// outcome read as a deficit growing from -0.9% to -7.6% across this range.
	// A single roughness would have caught either weakly or not at all.
	for (f32 const exitRoughness : { 0.1f, 0.5f, 0.9f }) {
		std::vector<GpuMorMaterial> const materials {
			make_material({ 0.8f, 0.8f, 0.8f }, { 0.0f, 0.0f, 0.0f }, 0.0f),
			make_material({ 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 6.0f),
			make_subsurface_material(
				{ 0.85f, 0.8f, 0.75f }, 0.3f, exitRoughness
			),
		};
		std::vector<MeshTri> tris;
		push_quad(tris, 8.0f, 0.0f, /*materialIndex=*/0u, /*faceDown=*/false);
		push_quad(tris, 1.0f, 3.0f, /*materialIndex=*/1u, /*faceDown=*/true);
		push_box(tris, { 0.0f, 0.6f, 0.0f }, 0.55f, /*materialIndex=*/2u);
		MorScene scene = build_scene(tris, materials);
		REQUIRE(scene.emitterLightCount == 1u);

		RenderConfig cfg {};
		cfg.sampleCount = test::statistical_samples(2048u);
		// the walk itself costs bounces, so a short path budget would clip the
		// exits this test exists to measure
		cfg.propagationDepth = 10u;
		srat::CameraOrbit cam = (
			overhead_camera((f32)cfg.width, (f32)cfg.height)
		);
		// tighter than the shared overhead framing: the box should fill the
		// frame, not sit as a speck in a field of floor
		cam.distance = 3.4f;
		cam.target = { 0.0f, 0.6f, 0.0f };

		RenderConfig on = cfg;
		on.meshNee = true;
		RenderConfig off = cfg;
		off.meshNee = false;
		MeanAndError const meanOn = render_mean(scene, cam, on, test::statistical_realizations(8u));
		MeanAndError const meanOff = render_mean(scene, cam, off, test::statistical_realizations(8u));

		CHECK(meanOn.nonFiniteCount == 0u);
		CHECK(meanOff.nonFiniteCount == 0u);
		REQUIRE(meanOff.mean > 0.0);

		f64 const delta = std::abs(meanOn.mean - meanOff.mean);
		f64 const sigma = (
			std::sqrt(
				meanOn.error * meanOn.error + meanOff.error * meanOff.error
			)
		);
		MESSAGE(
			"subsurface exit, exit roughness ", exitRoughness,
			": nee-on ", meanOn.mean, " vs nee-off ", meanOff.mean,
			"  delta ", delta / meanOff.mean * 100.0, "%",
			"  sigma ", sigma / meanOff.mean * 100.0, "%",
			"  = ", delta / std::max(sigma, 1e-12), " sigma"
		);
		CAPTURE(exitRoughness);
		CAPTURE(meanOn.mean);
		CAPTURE(meanOff.mean);
		CAPTURE(sigma);
		CHECK(delta < test::statistical_sigma_gate() * sigma);

		scene.destroy();
	}
}

TEST_CASE("emitter nee scene: on and off converge inside a scattering medium") {
	// tier 3.5. Homogeneous fog reaches the same medium-vertex nee block a
	// nanovdb grid does -- utilIrradianceWalk's scatter loop draws a free-flight
	// distance from each field and takes the nearer, then runs one shared nee
	// block whichever won. Fog is the half of that with no grid to author and a
	// closed-form transmittance, so a discrepancy here is the estimator's and
	// not delta tracking's.
	//
	// This is the case that catches a medium scatter misreporting its
	// light-sampling state: the scatter vertex runs mesh nee, so an emissive
	// triangle the phase-sampled continuation then hits must be balance-weighted
	// against that density, not taken at full radiance.
	MorScene scene = build_floor_and_emitter(1.0f, 3.0f, 6.0f);
	REQUIRE(scene.emitterLightCount > 0u);

	RenderConfig cfg {};
	cfg.sampleCount = test::statistical_samples(1024u);
	cfg.propagationDepth = 10u;
	// optical depth has to stay well under 1. fog applies over
	// min(segmentLength, fogDistanceMax) measured from each segment's start, so
	// an escaping ray sees the full cap -- at sigma 0.2 x 40 that is tau = 8,
	// where the scatter loop's hard 64-step cap truncates nee-on and nee-off by
	// different amounts and the comparison stops measuring the estimator.
	// 0.02 x 15 = 0.3 keeps single scattering dominant
	cfg.fogThickness = 0.02f;
	cfg.fogAlbedo = { 0.9f, 0.9f, 0.9f };
	cfg.fogDistanceMax = 15.0f;
	srat::CameraOrbit const cam = overhead_camera((f32)cfg.width, (f32)cfg.height);

	RenderConfig on = cfg;
	on.meshNee = true;
	RenderConfig off = cfg;
	off.meshNee = false;
	MeanAndError const meanOn = render_mean(scene, cam, on, test::statistical_realizations(4u));
	MeanAndError const meanOff = render_mean(scene, cam, off, test::statistical_realizations(4u));

	CHECK(meanOn.nonFiniteCount == 0u);
	CHECK(meanOff.nonFiniteCount == 0u);
	REQUIRE(meanOff.mean > 0.0);

	f64 const delta = std::abs(meanOn.mean - meanOff.mean);
	f64 const sigma = (
		std::sqrt(
			meanOn.error * meanOn.error + meanOff.error * meanOff.error
		)
	);
	MESSAGE(
		"fog medium: nee-on ", meanOn.mean, " vs nee-off ", meanOff.mean,
		"  delta ", delta / meanOff.mean * 100.0, "%",
		"  sigma ", sigma / meanOff.mean * 100.0, "%",
		"  = ", delta / std::max(sigma, 1e-12), " sigma"
	);
	CAPTURE(meanOn.mean);
	CAPTURE(meanOff.mean);
	CAPTURE(sigma);
	CHECK(delta < test::statistical_sigma_gate() * sigma);

	scene.destroy();
}

TEST_CASE("emitter nee scene: env nee inside fog, on vs off") {
	// discriminator, not a gate: the medium vertex's *environment* branch
	// predates P4 and has never been tested. If it shows the same on/off
	// discrepancy the mesh branch does, the defect is in the shared medium
	// structure rather than in anything P4 added.
	std::vector<GpuMorMaterial> const materials {
		make_material({ 0.8f, 0.8f, 0.8f }, { 0.0f, 0.0f, 0.0f }, 0.0f),
	};
	std::vector<MeshTri> tris;
	push_quad(tris, 8.0f, 0.0f, 0u, false);
	MorScene scene = build_scene(tris, materials);
	REQUIRE(scene.emitterLightCount == 0u);
	EnvScene env = make_env_sky(ENVIRONMENT_OUTPUT_DIR "emitter_nee_sky.exr");

	RenderConfig cfg {};
	cfg.sampleCount = test::statistical_samples(1024u);
	cfg.propagationDepth = 10u;
	cfg.envMapHandlesVa = env.va();
	cfg.meshNee = false;
	srat::CameraOrbit const cam = overhead_camera((f32)cfg.width, (f32)cfg.height);

	for (f32 const sigma : { 0.0f, 0.02f, 0.05f }) {
		RenderConfig on = cfg;
		on.fogThickness = sigma;
		on.fogAlbedo = { 0.9f, 0.9f, 0.9f };
		on.fogDistanceMax = 15.0f;
		on.envNee = true;
		RenderConfig off = on;
		off.envNee = false;
		MeanAndError const a = render_mean(scene, cam, on, test::statistical_realizations(4u));
		MeanAndError const b = render_mean(scene, cam, off, test::statistical_realizations(4u));
		f64 const delta = std::abs(a.mean - b.mean);
		f64 const sigmaErr = (
			std::sqrt(a.error * a.error + b.error * b.error)
		);
		MESSAGE(
			"fog sigma ", sigma, ": env-nee-on ", a.mean, " vs off ", b.mean,
			"  delta ", delta / b.mean * 100.0, "%",
			"  = ", delta / std::max(sigmaErr, 1e-12), " sigma"
		);
	}

	env.destroy();
	scene.destroy();
}

// ---------------------------------------------------------------------------
// tier 3.13 of plan-nee.md -- trap 1, the delta-bounce exclusion.
//
// A delta lobe sends the path along exactly one direction, which the light
// technique has zero probability of drawing. Its emission must therefore keep
// full weight. Balancing it against a nonzero lightPdf is what cull documented
// as "direct light seen straight through thin glass renders too dark", and the
// arithmetic is brutal rather than subtle: openPbrTransmissionPdf returns a
// literal 1.0 on its delta branch, so against an emitter density of order 2-3
// the balance weight lands near 0.3 and most of the light simply disappears.
//
// Every config puts a delta interface between the emitter and the receiver, so
// the shadow ray is blocked by that interface and nee genuinely cannot produce
// the path -- exactly the geometry the exclusion exists for.
// ---------------------------------------------------------------------------

namespace {

// the four ways a lobe becomes delta in this renderer. openPbrSampleWo's tir
// fallback is not among them: it draws a continuous h, so it is only delta when
// roughness is already 0, which skRough0Transmit covers
enum class DeltaKind {
	thinWalled,
	matchedIor,
	rough0Transmit,
	rough0Mirror,
};

// std::string, not char const *: doctest's MESSAGE stringifies a raw pointer
// as its address rather than its contents
std::string delta_kind_name(DeltaKind const kind) {
	switch (kind) {
		case DeltaKind::thinWalled: return "thin-walled transmission";
		case DeltaKind::matchedIor: return "matched-ior transmission";
		case DeltaKind::rough0Transmit: return "roughness-0 transmission";
		case DeltaKind::rough0Mirror: return "roughness-0 mirror";
	}
	return "?";
}

GpuMorMaterial make_delta_material(DeltaKind const kind) {
	GpuMorMaterial m = make_material({ 0.8f, 0.8f, 0.8f }, { 0.0f, 0.0f, 0.0f }, 0.0f);
	if (kind == DeltaKind::rough0Mirror) {
		m.baseColor = comp3({ 1.0f, 1.0f, 1.0f });
		m.baseMetalness = comp1(1.0f);
		m.specularWeight = comp1(1.0f);
		m.specularRoughness = comp1(0.0f);
		return m;
	}
	// a pure transmitter: no diffuse base competing for the lobe pick, so
	// essentially every path through the panel takes the delta branch
	m.baseWeight = comp1(0.0f);
	m.transmissionWeight = comp1(1.0f);
	m.specularWeight = comp1(1.0f);
	// deliberately nonzero, so thinWalled/matchedIor are delta for their own
	// reason rather than by way of the roughness clause
	m.specularRoughness = comp1(0.3f);
	switch (kind) {
		case DeltaKind::thinWalled:
			m.geometryThinWalled = comp1(1.0f);
			break;
		case DeltaKind::matchedIor:
			// openPbrEffectiveIor(ior, 0) == 1 exactly, so etaI == etaT at this
			// interface and both EvaluateF and Pdf take their delta branch
			m.specularWeight = comp1(0.0f);
			break;
		case DeltaKind::rough0Transmit:
			m.specularRoughness = comp1(0.0f);
			break;
		default:
			break;
	}
	return m;
}

} // namespace

TEST_CASE("emitter nee scene: delta lobes keep full weight on the emitter") {
	for (
		DeltaKind const kind : {
			DeltaKind::thinWalled,
			DeltaKind::matchedIor,
			DeltaKind::rough0Transmit,
			DeltaKind::rough0Mirror,
		}
	) {
		bool const isMirror = kind == DeltaKind::rough0Mirror;
		std::vector<GpuMorMaterial> const materials {
			isMirror
				? make_delta_material(kind)
				: make_material({ 0.8f, 0.8f, 0.8f }, { 0.0f, 0.0f, 0.0f }, 0.0f),
			make_material({ 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 6.0f),
			make_delta_material(kind),
		};
		std::vector<MeshTri> tris;
		push_quad(tris, 8.0f, 0.0f, /*materialIndex=*/0u, /*faceDown=*/false);
		push_quad(tris, 1.0f, 3.0f, /*materialIndex=*/1u, /*faceDown=*/true);
		if (!isMirror) {
			// a panel just under the emitter: wide enough to cover its whole
			// footprint, so no shadow ray from the floor reaches the emitter
			// without crossing it
			push_quad(tris, 1.6f, 2.2f, /*materialIndex=*/2u, /*faceDown=*/false);
		}
		MorScene scene = build_scene(tris, materials);
		REQUIRE(scene.emitterLightCount == 1u);

		RenderConfig cfg {};
		cfg.sampleCount = test::statistical_samples(2048u);
		cfg.propagationDepth = 8u;
		srat::CameraOrbit const cam = (
			overhead_camera((f32)cfg.width, (f32)cfg.height)
		);

		RenderConfig on = cfg;
		on.meshNee = true;
		RenderConfig off = cfg;
		off.meshNee = false;
		MeanAndError const meanOn = render_mean(scene, cam, on, test::statistical_realizations(6u));
		MeanAndError const meanOff = render_mean(scene, cam, off, test::statistical_realizations(6u));

		CHECK(meanOn.nonFiniteCount == 0u);
		CHECK(meanOff.nonFiniteCount == 0u);
		REQUIRE(meanOff.mean > 0.0);

		f64 const delta = std::abs(meanOn.mean - meanOff.mean);
		f64 const sigma = (
			std::sqrt(
				meanOn.error * meanOn.error + meanOff.error * meanOff.error
			)
		);
		MESSAGE(
			delta_kind_name(kind),
			": nee-on ", meanOn.mean, " vs nee-off ", meanOff.mean,
			"  delta ", delta / meanOff.mean * 100.0, "%",
			"  sigma ", sigma / meanOff.mean * 100.0, "%",
			"  = ", delta / std::max(sigma, 1e-12), " sigma"
		);
		CAPTURE(meanOn.mean);
		CAPTURE(meanOff.mean);
		CAPTURE(sigma);
		CHECK(delta < test::statistical_sigma_gate() * sigma);

		scene.destroy();
	}
}

TEST_CASE("emitter nee scene: emission is two-sided") {
	// tier 3.7, trap 5. Decision 2 of plan-nee.md chose two-sided emission, and
	// the thing that actually has to hold is that the sampler, the pdf and the
	// bsdf-hit branch all agree about it -- Blender #115997 is this exact class
	// of bug surviving in a mature renderer.
	//
	// Flipping the emitter quad's winding is the sharpest way to ask: under
	// two-sided emission the two renders are the same scene, so any difference
	// is one of the three halves disagreeing. It also cannot be satisfied by
	// accident, the way "does the floor receive light" can.
	std::vector<MeanAndError> windingMeans;
	for (bool const faceDown : { true, false }) {
		std::vector<GpuMorMaterial> const materials {
			make_material({ 0.8f, 0.8f, 0.8f }, { 0.0f, 0.0f, 0.0f }, 0.0f),
			make_material({ 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 4.0f),
		};
		std::vector<MeshTri> tris;
		push_quad(tris, 8.0f, 0.0f, /*materialIndex=*/0u, /*faceDown=*/false);
		push_quad(tris, 1.0f, 3.0f, /*materialIndex=*/1u, faceDown);
		MorScene scene = build_scene(tris, materials);
		REQUIRE(scene.emitterLightCount == 1u);

		RenderConfig cfg {};
		cfg.sampleCount = test::statistical_samples(1024u);
		srat::CameraOrbit const cam = (
			overhead_camera((f32)cfg.width, (f32)cfg.height)
		);
		RenderConfig on = cfg;
		on.meshNee = true;
		RenderConfig off = cfg;
		off.meshNee = false;
		MeanAndError const meanOn = render_mean(scene, cam, on, test::statistical_realizations(4u));
		MeanAndError const meanOff = render_mean(scene, cam, off, test::statistical_realizations(4u));

		f64 const delta = std::abs(meanOn.mean - meanOff.mean);
		f64 const sigma = (
			std::sqrt(meanOn.error * meanOn.error + meanOff.error * meanOff.error)
		);
		MESSAGE(
			"emitter faceDown=", faceDown,
			": nee-on ", meanOn.mean, " vs nee-off ", meanOff.mean,
			"  delta ", delta / meanOff.mean * 100.0, "%",
			"  = ", delta / std::max(sigma, 1e-12), " sigma"
		);
		CAPTURE(faceDown);
		CHECK(meanOn.nonFiniteCount == 0u);
		REQUIRE(meanOff.mean > 0.0);
		CHECK(delta < test::statistical_sigma_gate() * sigma);
		windingMeans.emplace_back(meanOn);

		scene.destroy();
	}

	// the winding must not matter at all under a two-sided convention: an
	// emitter whose normals point away from the floor lights it identically.
	// compared against each other rather than a hand-written constant, so the
	// gate stays meaningful if the scene is ever retuned
	REQUIRE(windingMeans.size() == 2u);
	f64 const windingDelta = (
		std::abs(windingMeans[0].mean - windingMeans[1].mean)
	);
	f64 const windingSigma = (
		std::sqrt(
			windingMeans[0].error * windingMeans[0].error
			+ windingMeans[1].error * windingMeans[1].error
		)
	);
	MESSAGE(
		"winding flip: ", windingMeans[0].mean, " vs ", windingMeans[1].mean,
		"  delta ", windingDelta / windingMeans[1].mean * 100.0, "%",
		"  = ", windingDelta / std::max(windingSigma, 1e-12), " sigma"
	);
	CHECK(windingDelta < test::statistical_sigma_gate() * windingSigma);
}

TEST_CASE("emitter nee scene: a clipped light list stays unbiased") {
	// tier 3.10, trap 7. The list carries no geometry, so an emitter missing
	// from it is not missing from the render: utilEmitterHitPdf returns 0 for a
	// triangle it cannot find, misBalanceWeight(bsdfPdf, 0) == 1, and the bsdf
	// technique carries that emitter at full weight. Clipping is therefore a
	// variance cost and never a correctness one -- which is the entire licence
	// for capping the list at all, so it is worth asserting rather than
	// assuming.
	//
	// The cap is applied at build time rather than by swapping the uploaded
	// buffer, so each model's emitterLightOffset/Count still describes the list
	// the shader indexes.
	std::vector<GpuMorMaterial> const materials {
		make_material({ 0.8f, 0.8f, 0.8f }, { 0.0f, 0.0f, 0.0f }, 0.0f),
		make_material({ 0.0f, 0.0f, 0.0f }, { 1.0f, 0.9f, 0.8f }, 5.0f),
		make_material({ 0.0f, 0.0f, 0.0f }, { 0.8f, 0.9f, 1.0f }, 5.0f),
	};
	std::vector<MeshTri> tris;
	push_quad(tris, 8.0f, 0.0f, /*materialIndex=*/0u, /*faceDown=*/false);
	// two separated emitters, so dropping one really removes a light the other
	// cannot stand in for
	{
		std::vector<MeshTri> a;
		push_quad(a, 0.7f, 3.0f, 1u, /*faceDown=*/true);
		for (MeshTri & t : a) {
			t.a.x -= 1.6f; t.b.x -= 1.6f; t.c.x -= 1.6f;
			tris.emplace_back(t);
		}
		std::vector<MeshTri> b;
		push_quad(b, 0.7f, 3.0f, 2u, /*faceDown=*/true);
		for (MeshTri & t : b) {
			t.a.x += 1.6f; t.b.x += 1.6f; t.c.x += 1.6f;
			tris.emplace_back(t);
		}
	}

	RenderConfig cfg {};
	cfg.sampleCount = test::statistical_samples(2048u);
	srat::CameraOrbit const cam = overhead_camera((f32)cfg.width, (f32)cfg.height);

	struct Leg {
		std::string label;
		u32 maxLights;
		bool meshNee;
	};
	std::vector<Leg> const legs {
		{ "full list", 4096u, true },
		{ "clipped to 1", 1u, true },
		{ "no list at all", 4096u, false },
	};

	std::vector<MeanAndError> results;
	for (Leg const & leg : legs) {
		MorScene scene = build_scene(tris, materials, leg.maxLights);
		if (leg.maxLights == 1u) {
			// the cap really did drop one, or this leg proves nothing
			REQUIRE(scene.emitterLightCount == 1u);
		}
		RenderConfig r = cfg;
		r.meshNee = leg.meshNee;
		results.emplace_back(render_mean(scene, cam, r, test::statistical_realizations(4u)));
		MESSAGE(
			leg.label, ": mean ", results.back().mean,
			"  sigma ", results.back().error / results.back().mean * 100.0, "%"
		);
		scene.destroy();
	}

	REQUIRE(results[0].mean > 0.0);
	for (size_t i = 1u; i < results.size(); ++i) {
		f64 const delta = std::abs(results[i].mean - results[0].mean);
		f64 const sigma = (
			std::sqrt(
				results[i].error * results[i].error
				+ results[0].error * results[0].error
			)
		);
		CAPTURE(i);
		CHECK(delta < test::statistical_sigma_gate() * sigma);
	}
}

TEST_CASE("emitter nee scene: an alpha-blend occluder stays unbiased") {
	// tier 3.12. utilTraceRay resolves BLEND coverage stochastically against
	// the path's rng stream, shadow rays included, so a half-covered panel
	// passes an nee connection half the time rather than thresholding. That is
	// unbiased only if the bsdf technique sees the same coverage distribution
	// through the same panel, which is what this compares.
	std::vector<GpuMorMaterial> materials {
		make_material({ 0.8f, 0.8f, 0.8f }, { 0.0f, 0.0f, 0.0f }, 0.0f),
		make_material({ 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 6.0f),
		make_material({ 0.7f, 0.7f, 0.7f }, { 0.0f, 0.0f, 0.0f }, 0.0f),
	};
	materials[2].alphaMode = MOR_ALPHA_MODE_BLEND;
	materials[2].geometryOpacity = comp1(0.5f);

	std::vector<MeshTri> tris;
	push_quad(tris, 8.0f, 0.0f, /*materialIndex=*/0u, /*faceDown=*/false);
	push_quad(tris, 1.0f, 3.0f, /*materialIndex=*/1u, /*faceDown=*/true);
	// wider than the emitter, so every connection from the floor crosses it
	push_quad(tris, 2.0f, 2.2f, /*materialIndex=*/2u, /*faceDown=*/false);
	MorScene scene = build_scene(tris, materials);
	REQUIRE(scene.emitterLightCount == 1u);

	RenderConfig cfg {};
	cfg.sampleCount = test::statistical_samples(2048u);
	srat::CameraOrbit const cam = overhead_camera((f32)cfg.width, (f32)cfg.height);
	RenderConfig on = cfg;
	on.meshNee = true;
	RenderConfig off = cfg;
	off.meshNee = false;
	MeanAndError const meanOn = render_mean(scene, cam, on, test::statistical_realizations(6u));
	MeanAndError const meanOff = render_mean(scene, cam, off, test::statistical_realizations(6u));

	f64 const delta = std::abs(meanOn.mean - meanOff.mean);
	f64 const sigma = (
		std::sqrt(meanOn.error * meanOn.error + meanOff.error * meanOff.error)
	);
	MESSAGE(
		"alpha-blend occluder: nee-on ", meanOn.mean, " vs nee-off ", meanOff.mean,
		"  delta ", delta / meanOff.mean * 100.0, "%",
		"  = ", delta / std::max(sigma, 1e-12), " sigma"
	);
	CHECK(meanOn.nonFiniteCount == 0u);
	REQUIRE(meanOff.mean > 0.0);
	CHECK(delta < test::statistical_sigma_gate() * sigma);

	scene.destroy();
}

namespace {

// an emissive texture with almost all of its energy in a small bright spot.
// deliberately pathological for the emitter list's `power`, which uses the
// material factors alone and so appraises this emitter by its average -- the
// point of the test is that the mis-appraisal costs variance and not bias
struct EmissiveTexture {
	vkof::Image image {};
	vkof::Sampler sampler {};
	u32 handle = 0u;

	void destroy() {
		if (sampler.id != 0u) { vkof::sampler_destroy(sampler); }
		if (image.id != 0u) { vkof::image_destroy(image); }
	}
};

EmissiveTexture make_spot_emissive_texture() {
	constexpr u32 skSize = 32u;
	std::vector<u8> pixels((size_t)skSize * skSize * 4u, 0u);
	for (u32 y = 0u; y < skSize; ++y) {
		for (u32 x = 0u; x < skSize; ++x) {
			size_t const i = ((size_t)y * skSize + x) * 4u;
			// a 4x4 white spot in an otherwise near-black texture: ~1.5% of
			// the area carries essentially all of the emitted energy
			bool const inSpot = (x >= 20u && x < 24u && y >= 8u && y < 12u);
			u8 const v = inSpot ? 255u : 4u;
			pixels[i + 0] = v;
			pixels[i + 1] = v;
			pixels[i + 2] = v;
			pixels[i + 3] = 255u;
		}
	}
	EmissiveTexture out {};
	out.image = vkof::image_create({
		.width = skSize,
		.height = skSize,
		.format = vkof::ImageFormat::r8g8b8a8_unorm,
		.mipLevels = 1,
		.optInitialData = srat::slice<u8 const>(pixels.data(), pixels.size()),
	});
	out.sampler = vkof::sampler_create({
		.magFilter = vkof::SamplerFilter::nearest,
		.minFilter = vkof::SamplerFilter::nearest,
		.addressModeU = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeV = vkof::SamplerAddressMode::clamp_to_edge,
		.addressModeW = vkof::SamplerAddressMode::clamp_to_edge,
	});
	out.handle = (
		vkof::image_sampler_handle({ .image = out.image, .sampler = out.sampler })
	);
	return out;
}

} // namespace

TEST_CASE("emitter nee scene: a textured emitter is steered, not re-coloured") {
	// tier 3.9. plan-nee.md section 3.2 deliberately diverges from cull here:
	// the emitter list's `power` ignores emissive textures and uses the
	// material factors alone, but the *radiance* at a sampled point is read
	// back through openPbrLoadMaterialDeriv at that point's uv, exactly as the
	// bsdf-hit branch does. So the texture is allowed to make the pick a bad
	// one -- it must never make the answer a wrong one.
	//
	// The texture concentrates ~all of its energy into ~1.5% of the quad,
	// which is the worst case for a factor-only appraisal: nee spreads its
	// samples uniformly over an emitter that is almost entirely black.
	EmissiveTexture texture = make_spot_emissive_texture();
	REQUIRE(texture.handle != 0u);

	std::vector<GpuMorMaterial> materials {
		make_material({ 0.8f, 0.8f, 0.8f }, { 0.0f, 0.0f, 0.0f }, 0.0f),
		make_material({ 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 40.0f),
	};
	materials[1].emissionColor.texture = texture.handle;

	std::vector<MeshTri> tris;
	push_quad(tris, 8.0f, 0.0f, /*materialIndex=*/0u, /*faceDown=*/false);
	push_quad(tris, 1.0f, 3.0f, /*materialIndex=*/1u, /*faceDown=*/true);
	MorScene scene = build_scene(tris, materials);
	REQUIRE(scene.emitterLightCount == 1u);

	RenderConfig cfg {};
	cfg.sampleCount = test::statistical_samples(4096u);
	srat::CameraOrbit const cam = overhead_camera((f32)cfg.width, (f32)cfg.height);
	RenderConfig on = cfg;
	on.meshNee = true;
	RenderConfig off = cfg;
	off.meshNee = false;
	MeanAndError const meanOn = (
		render_mean(scene, cam, on, test::statistical_realizations(6u))
	);
	MeanAndError const meanOff = (
		render_mean(scene, cam, off, test::statistical_realizations(6u))
	);

	f64 const delta = std::abs(meanOn.mean - meanOff.mean);
	f64 const sigma = (
		std::sqrt(meanOn.error * meanOn.error + meanOff.error * meanOff.error)
	);
	MESSAGE(
		"textured emitter: nee-on ", meanOn.mean, " vs nee-off ", meanOff.mean,
		"  delta ", delta / meanOff.mean * 100.0, "%",
		"  = ", delta / std::max(sigma, 1e-12), " sigma"
	);
	CHECK(meanOn.nonFiniteCount == 0u);
	CHECK(meanOff.nonFiniteCount == 0u);
	REQUIRE(meanOff.mean > 0.0);
	CHECK(delta < test::statistical_sigma_gate() * sigma);

	scene.destroy();
	texture.destroy();
}

namespace {

// -- instanced scene -------------------------------------------------------

// two blases (a static floor, and an emitter quad instanced N times) and one
// model-indirect entry per tlas instance. build_scene deliberately stays
// single-instance -- fifteen tests depend on its exact shape -- so the
// instancing case gets its own builder rather than a parameter on that one.
//
// each instance carries its OWN materials buffer over shared geometry, which
// is what lets a leg light exactly one instance while every instance stays
// present as geometry. Transport is linear in the sources only when the
// occluders are identical across the comparison.
struct InstancedScene {
	std::vector<vkof::Buffer> buffers;
	vkof::Buffer models {};
	vkof::Buffer emitterLights {};
	u32 emitterLightCount = 0u;
	vkof::AccelerationStructureBlas floorBlas {};
	vkof::AccelerationStructureBlas quadBlas {};
	vkof::AccelerationStructureTlas tlas {};

	void destroy() {
		if (tlas.id != 0u) { vkof::tlas_destroy(tlas); }
		if (quadBlas.id != 0u) { vkof::blas_destroy(quadBlas); }
		if (floorBlas.id != 0u) { vkof::blas_destroy(floorBlas); }
		if (emitterLights.id != 0u) { vkof::buffer_destroy(emitterLights); }
		if (models.id != 0u) { vkof::buffer_destroy(models); }
		for (vkof::Buffer & b : buffers) { vkof::buffer_destroy(b); }
	}
};

// geometry + attribute buffers for one standalone mesh, shared by every
// instance that references it
struct MeshBuffers {
	u64 positions;
	u64 attributes;
	u64 flatIndices;
	u64 flatMeshlets;
	u64 meshlets;
	u64 uvTransforms;
	u32 triangleCount;
	u32 vertexCount;
	// object-space area of the single meshlet, for the emitter build
	f32 area;
};

MeshBuffers upload_mesh(
	std::vector<MeshTri> const & tris,
	std::vector<vkof::Buffer> & owned
) {
	std::vector<f32v3> positions;
	std::vector<GpuMorVertexAttribute> attributes;
	std::vector<u32> flatIndices;
	std::vector<u32> flatMeshlets;
	f32 area = 0.0f;
	for (MeshTri const & tri : tris) {
		u32 const base = (u32)positions.size();
		f32v3 const geoNormal = (
			f32v3_normalize(f32v3_cross(tri.b - tri.a, tri.c - tri.a))
		);
		area += (
			0.5f * f32v3_length(f32v3_cross(tri.b - tri.a, tri.c - tri.a))
		);
		f32v3 const corners[3] = { tri.a, tri.b, tri.c };
		f32v2 const cornerUvs[3] = { tri.uvA, tri.uvB, tri.uvC };
		for (u32 c = 0u; c < 3u; ++c) {
			positions.emplace_back(corners[c]);
			attributes.emplace_back(GpuMorVertexAttribute {
				.normal = geoNormal,
				.uv = cornerUvs[c],
				.tangent = { 1.0f, 0.0f, 0.0f, 1.0f },
			});
		}
		flatIndices.emplace_back(base + 0u);
		flatIndices.emplace_back(base + 1u);
		flatIndices.emplace_back(base + 2u);
		flatMeshlets.emplace_back(0u);
	}
	std::vector<GpuMorMeshlet> const meshlets {
		GpuMorMeshlet {
			.vertexOffset = 0u,
			.vertexCount = 0u,
			.triangleOffset = 0u,
			.triangleCount = (u32)tris.size(),
			.instanceIndex = 0u,
			.materialIndex = 0u,
		},
	};
	std::vector<GpuMorUvTransform> const uvTransforms {
		GpuMorUvTransform {
			.basisU = { 1.0f, 0.0f },
			.basisV = { 0.0f, 1.0f },
			.offset = { 0.0f, 0.0f },
		},
	};
	auto upload = [&](void const * data, u64 const bytes) {
		vkof::Buffer const b = test::upload_bytes(data, bytes);
		owned.emplace_back(b);
		return vkof::buffer_virtual_address(b);
	};
	return MeshBuffers {
		.positions = upload(positions.data(), positions.size() * sizeof(f32v3)),
		.attributes = upload(
			attributes.data(),
			attributes.size() * sizeof(GpuMorVertexAttribute)
		),
		.flatIndices = upload(
			flatIndices.data(), flatIndices.size() * sizeof(u32)
		),
		.flatMeshlets = upload(
			flatMeshlets.data(), flatMeshlets.size() * sizeof(u32)
		),
		.meshlets = upload(
			meshlets.data(), meshlets.size() * sizeof(GpuMorMeshlet)
		),
		.uvTransforms = upload(
			uvTransforms.data(), uvTransforms.size() * sizeof(GpuMorUvTransform)
		),
		.triangleCount = (u32)tris.size(),
		.vertexCount = (u32)positions.size(),
		.area = area,
	};
}

// emissiveMask[i] == true lights instance i; the others stay present as black
// geometry so every leg of the comparison sees identical occluders
InstancedScene build_instanced_emitter_scene(
	std::vector<f32m44> const & instanceTransforms,
	std::vector<bool> const & emissiveMask,
	f32 const emissionLuminance
) {
	REQUIRE(instanceTransforms.size() == emissiveMask.size());
	InstancedScene scene {};

	std::vector<MeshTri> floorTris;
	push_quad(floorTris, 8.0f, 0.0f, /*materialIndex=*/0u, /*faceDown=*/false);
	std::vector<MeshTri> quadTris;
	push_quad(quadTris, 0.5f, 0.0f, /*materialIndex=*/0u, /*faceDown=*/true);

	MeshBuffers const floorMesh = upload_mesh(floorTris, scene.buffers);
	MeshBuffers const quadMesh = upload_mesh(quadTris, scene.buffers);

	auto uploadOwned = [&](void const * data, u64 const bytes) {
		vkof::Buffer const b = test::upload_bytes(data, bytes);
		scene.buffers.emplace_back(b);
		return vkof::buffer_virtual_address(b);
	};

	std::vector<GpuMorMaterial> const floorMaterials {
		make_material({ 0.8f, 0.8f, 0.8f }, { 0.0f, 0.0f, 0.0f }, 0.0f),
	};
	u64 const floorMaterialsVa = uploadOwned(
		floorMaterials.data(), floorMaterials.size() * sizeof(GpuMorMaterial)
	);

	// model 0 is the floor; models 1..N are the emitter instances
	std::vector<GpuResolveModelIndirect> models;
	GpuResolveModelIndirect floorModel {};
	floorModel.meshlets = floorMesh.meshlets;
	floorModel.materials = floorMaterialsVa;
	floorModel.uvTransforms = floorMesh.uvTransforms;
	floorModel.positions = floorMesh.positions;
	floorModel.attributes = floorMesh.attributes;
	floorModel.flatIndices = floorMesh.flatIndices;
	floorModel.flatMeshlets = floorMesh.flatMeshlets;
	floorModel.modelMatrix = f32m44_identity();
	floorModel.emitterLightOffset = 0u;
	floorModel.emitterLightCount = 0u;
	models.emplace_back(floorModel);

	std::vector<GpuEmitterLight> lights;
	for (usize i = 0u; i < instanceTransforms.size(); ++i) {
		// a private materials buffer per instance over shared geometry
		std::vector<GpuMorMaterial> const instanceMaterials {
			make_material(
				{ 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f },
				emissiveMask[i] ? emissionLuminance : 0.0f
			),
		};
		u64 const materialsVa = uploadOwned(
			instanceMaterials.data(),
			instanceMaterials.size() * sizeof(GpuMorMaterial)
		);
		GpuResolveModelIndirect model {};
		model.meshlets = quadMesh.meshlets;
		model.materials = materialsVa;
		model.uvTransforms = quadMesh.uvTransforms;
		model.positions = quadMesh.positions;
		model.attributes = quadMesh.attributes;
		model.flatIndices = quadMesh.flatIndices;
		model.flatMeshlets = quadMesh.flatMeshlets;
		model.modelMatrix = instanceTransforms[i];
		u32 const drawIndex = (u32)models.size();
		ponder::EmitterRange range { (u32)lights.size(), 0u, 0u };
		if (emissiveMask[i]) {
			std::vector<ponder::EmitterMeshletInput> const inputs {
				ponder::EmitterMeshletInput {
					.primitiveIndexBase = 0u,
					.triangleCount = quadMesh.triangleCount,
					.objectArea = quadMesh.area,
					.emissionColor = { 1.0f, 1.0f, 1.0f },
					.emissionLuminance = emissionLuminance,
				},
			};
			range = ponder::emitter_lights_append(
				lights,
				srat::slice<ponder::EmitterMeshletInput const>(
					inputs.data(), inputs.size()
				),
				instanceTransforms[i], drawIndex, /*maxLights=*/4096u
			);
		}
		model.emitterLightOffset = range.offset;
		model.emitterLightCount = range.count;
		models.emplace_back(model);
	}
	(void)ponder::emitter_lights_finalize(lights);
	scene.emitterLightCount = (u32)lights.size();
	if (!lights.empty()) {
		scene.emitterLights = test::upload_bytes(
			lights.data(), lights.size() * sizeof(GpuEmitterLight)
		);
	}
	scene.models = test::upload_bytes(
		models.data(), models.size() * sizeof(GpuResolveModelIndirect)
	);

	scene.floorBlas = vkof::blas_create({
		.positionVa = floorMesh.positions,
		.vertexCount = floorMesh.vertexCount,
		.indexVa = floorMesh.flatIndices,
		.triangleCount = floorMesh.triangleCount,
		.isOpaque = true,
	});
	scene.quadBlas = vkof::blas_create({
		.positionVa = quadMesh.positions,
		.vertexCount = quadMesh.vertexCount,
		.indexVa = quadMesh.flatIndices,
		.triangleCount = quadMesh.triangleCount,
		.isOpaque = true,
	});
	REQUIRE(scene.floorBlas.id != 0u);
	REQUIRE(scene.quadBlas.id != 0u);

	std::vector<vkof::TlasInstance> instances;
	instances.emplace_back(vkof::TlasInstance {
		.blas = scene.floorBlas,
		.transform = f32m44_identity(),
		.instanceCustomIndex = 0u,
		.rayMask = 0xFFu,
	});
	for (usize i = 0u; i < instanceTransforms.size(); ++i) {
		instances.emplace_back(vkof::TlasInstance {
			.blas = scene.quadBlas,
			.transform = instanceTransforms[i],
			// must equal the models-indirect slot: the shader resolves a hit's
			// model through instanceCustomIndex
			.instanceCustomIndex = (u32)(i + 1u),
			.rayMask = 0xFFu,
		});
	}
	scene.tlas = vkof::tlas_create({ .maxInstances = (u32)instances.size() });
	REQUIRE(scene.tlas.id != 0u);

	vkof::RenderNode const node = (
		vkof::render_node_create({ .queue = vkof::CommandQueue::compute })
	);
	vkof::render_node_callback({
		.node = node,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			vkof::tlas_build(
				cmd, scene.tlas,
				srat::slice<vkof::TlasInstance const>(
					instances.data(), instances.size()
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

} // namespace

TEST_CASE("emitter nee scene: instanced emitters sum linearly") {
	// tier 3.8, trap 7. Instances share a blas but not a world area, so the
	// emitter build has to apply each instance's own transform: the same
	// meshlet at 1.6x scale is 2.56x the power, and the sampler has to place
	// its sampled point in that instance's world space, not the shared object
	// space. Getting either wrong produces a plausible image that is simply
	// the wrong brightness.
	//
	// Transport is linear in the emitters, so lighting all three at once must
	// equal the sum of lighting each alone -- with all three present as
	// geometry in every leg, so the occluders never change. A deliberately
	// non-uniform set of transforms: one identity, one uniformly scaled, one
	// scaled per-axis (where emitter_transform_area_scale is only an
	// |det|^(2/3) estimate and so the most likely to drift).
	// f32m44 is a flat column-major array: m[col*4 + row], so the diagonal is
	// 0/5/10 and the translation column is 12/13/14
	std::vector<f32m44> transforms;
	{
		f32m44 a = f32m44_identity();
		a.m[12] = -2.2f;
		a.m[13] = 3.0f;
		transforms.emplace_back(a);
		f32m44 b = f32m44_identity();
		b.m[0] = 1.6f;
		b.m[5] = 1.6f;
		b.m[10] = 1.6f;
		b.m[13] = 3.4f;
		transforms.emplace_back(b);
		f32m44 c = f32m44_identity();
		c.m[0] = 1.7f;
		c.m[5] = 1.0f;
		c.m[10] = 0.6f;
		c.m[12] = 2.4f;
		c.m[13] = 3.0f;
		transforms.emplace_back(c);
	}

	RenderConfig cfg {};
	cfg.sampleCount = test::statistical_samples(2048u);
	cfg.meshNee = true;
	srat::CameraOrbit const cam = overhead_camera((f32)cfg.width, (f32)cfg.height);
	u32 const realizations = test::statistical_realizations(6u);
	f32 const luminance = 8.0f;

	InstancedScene all = (
		build_instanced_emitter_scene(transforms, { true, true, true }, luminance)
	);
	REQUIRE(all.emitterLightCount == 3u);
	MeanAndError const combined = render_mean(all, cam, cfg, realizations);
	all.destroy();

	f64 sumMean = 0.0;
	f64 sumVariance = 0.0;
	for (usize i = 0u; i < transforms.size(); ++i) {
		std::vector<bool> mask(transforms.size(), false);
		mask[i] = true;
		InstancedScene one = (
			build_instanced_emitter_scene(transforms, mask, luminance)
		);
		REQUIRE(one.emitterLightCount == 1u);
		MeanAndError const single = render_mean(one, cam, cfg, realizations);
		MESSAGE("instance ", i, " alone: mean ", single.mean);
		sumMean += single.mean;
		sumVariance += single.error * single.error;
		CHECK(single.nonFiniteCount == 0u);
		one.destroy();
	}

	f64 const delta = std::abs(combined.mean - sumMean);
	f64 const sigma = (
		std::sqrt(combined.error * combined.error + sumVariance)
	);
	MESSAGE(
		"instanced emitters: combined ", combined.mean, " vs sum ", sumMean,
		"  delta ", delta / sumMean * 100.0, "%",
		"  = ", delta / std::max(sigma, 1e-12), " sigma"
	);
	CHECK(combined.nonFiniteCount == 0u);
	REQUIRE(sumMean > 0.0);
	CHECK(delta < test::statistical_sigma_gate() * sigma);
}

// ---------------------------------------------------------------------------
// -- dispersion
// ---------------------------------------------------------------------------
namespace {

// glossy glass floor under the emitter: the camera sees only its fresnel
// reflection, which depends on n(λ)
MorScene build_dispersive_floor(
	f32 const dispersionScale,
	f32 const abbeNumber
) {
	GpuMorMaterial glass = (
		make_material({ 0.8f, 0.8f, 0.8f }, { 0.0f, 0.0f, 0.0f }, 0.0f)
	);
	glass.baseWeight = comp1(0.0f);
	glass.transmissionWeight = comp1(1.0f);
	glass.specularWeight = comp1(1.0f);
	glass.specularRoughness = comp1(0.2f);
	glass.specularIor = comp1(1.5f);
	glass.transmissionDispersionScale = comp1(dispersionScale);
	glass.transmissionDispersionAbbeNumber = comp1(abbeNumber);
	std::vector<GpuMorMaterial> const materials {
		glass,
		make_material({ 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 4.0f),
	};
	std::vector<MeshTri> tris;
	push_quad(tris, 8.0f, 0.0f, /*materialIndex=*/0u, /*faceDown=*/false);
	push_quad(tris, 1.0f, 3.0f, /*materialIndex=*/1u, /*faceDown=*/true);
	return build_scene(tris, materials);
}

struct DispersionSummary {
	f64 x;
	f64 y;
	f64 z;
	// pixel-to-pixel standard deviation of z / y
	f64 chromaSpread;
};

DispersionSummary summarize_dispersion(
	std::vector<PixelXyz> const & pixels,
	std::vector<bool> const & mask
) {
	DispersionSummary summary { 0.0, 0.0, 0.0, 0.0 };
	f64 ratioSum = 0.0;
	f64 ratioSqSum = 0.0;
	u32 count = 0u;
	for (u32 i = 0u; i < (u32)pixels.size(); ++i) {
		if (!mask[i]) { continue; }
		PixelXyz const & p = pixels[i];
		summary.x += p.x;
		summary.y += p.y;
		summary.z += p.z;
		f64 const ratio = p.z / p.y;
		ratioSum += ratio;
		ratioSqSum += ratio * ratio;
		++count;
	}
	REQUIRE(count > 0u);
	f64 const mean = ratioSum / (f64)count;
	summary.chromaSpread = (
		std::sqrt(std::max(0.0, ratioSqSum / (f64)count - mean * mean))
	);
	return summary;
}

// floor pixels the glass reflection lights; the emitter itself is far
// brighter and never touches the glass
std::vector<bool> dispersion_floor_mask(
	WavefrontView const & view,
	WavefrontRenderConfig const & cfg,
	std::vector<PixelXyz> const & reference
) {
	u32 const pixelCount = cfg.width * cfg.height;
	std::vector<bool> isFloor(pixelCount, false);
	for (u32 y = 0u; y < cfg.height; ++y) {
		for (u32 x = 0u; x < cfg.width; ++x) {
			f32v3 const dir = (
				wavefront_ray_direction(
					view, (f32)x + 0.5f, (f32)y + 0.5f, cfg.width, cfg.height
				)
			);
			if (dir.y >= -1e-6f) { continue; }
			f32 const tEmitter = (3.0f - view.position.y) / dir.y;
			f32 const ex = view.position.x + dir.x * tEmitter;
			f32 const ez = view.position.z + dir.z * tEmitter;
			bool const hitsEmitter = (
				tEmitter > 0.0f
				&& std::abs(ex) <= 1.05f
				&& std::abs(ez) <= 1.05f
			);
			isFloor[y * cfg.width + x] = !hitsEmitter;
		}
	}
	f64 maxY = 0.0;
	for (u32 i = 0u; i < pixelCount; ++i) {
		if (isFloor[i]) { maxY = std::max(maxY, reference[i].y); }
	}
	REQUIRE(maxY > 0.0);
	std::vector<bool> mask(pixelCount, false);
	u32 litCount = 0u;
	for (u32 i = 0u; i < pixelCount; ++i) {
		mask[i] = isFloor[i] && reference[i].y > 0.2 * maxY;
		litCount += mask[i] ? 1u : 0u;
	}
	REQUIRE(litCount > 50u);
	return mask;
}

WavefrontView dispersion_view(WavefrontRenderConfig const & cfg) {
	return (
		wavefront_view_from_orbit(
			overhead_camera((f32)cfg.width, (f32)cfg.height)
		)
	);
}

} // namespace

TEST_CASE("wavefront dispersion: hero wavelength termination and rgb fallback") {
	WavefrontRenderConfig cfg {};
	cfg.sampleCount = test::statistical_samples(2048u);
	WavefrontView const view = dispersion_view(cfg);

	// -- renders: no dispersion, negligible dispersion (V = 1e5), so
	// termination runs but n(λ) stays n_d
	MorScene sceneNone = build_dispersive_floor(0.0f, 20.0f);
	MorScene sceneTiny = build_dispersive_floor(1.0f, 1.0e5f);
	MorScene sceneStrong = build_dispersive_floor(4.0f, 20.0f);
	WavefrontRenderConfig spectralCfg = cfg;
	spectralCfg.spectral = true;
	std::vector<PixelXyz> const spectralNone = (
		render_scene_wavefront_xyz(sceneNone, view, spectralCfg)
	);
	std::vector<PixelXyz> const spectralTiny = (
		render_scene_wavefront_xyz(sceneTiny, view, spectralCfg)
	);
	std::vector<PixelXyz> const rgbNone = (
		render_scene_wavefront_xyz(sceneNone, view, cfg)
	);
	std::vector<PixelXyz> const rgbStrong = (
		render_scene_wavefront_xyz(sceneStrong, view, cfg)
	);
	std::vector<bool> const mask = (
		dispersion_floor_mask(view, cfg, spectralNone)
	);
	DispersionSummary const none = summarize_dispersion(spectralNone, mask);
	DispersionSummary const tiny = summarize_dispersion(spectralTiny, mask);

	// -- hero-only estimator: unbiased, but noisier in chroma
	f64 const tinyDx = std::abs(tiny.x / none.x - 1.0);
	f64 const tinyDy = std::abs(tiny.y / none.y - 1.0);
	f64 const tinyDz = std::abs(tiny.z / none.z - 1.0);
	f64 const spreadRatio = tiny.chromaSpread / none.chromaSpread;
	MESSAGE(
		"negligible dispersion xyz delta ",
		tinyDx * 100.0, "% ", tinyDy * 100.0, "% ", tinyDz * 100.0, "%",
		"  chroma spread x", spreadRatio
	);
	CHECK(tinyDx < 0.02);
	CHECK(tinyDy < 0.02);
	CHECK(tinyDz < 0.02);
	CHECK(spreadRatio > 1.5);

	// -- rgb ignores dispersion
	f64 maxY = 0.0;
	f64 rgbMaxDiff = 0.0;
	for (u32 i = 0u; i < (u32)rgbNone.size(); ++i) {
		maxY = std::max(maxY, rgbNone[i].y);
		rgbMaxDiff = std::max({
			rgbMaxDiff,
			std::abs(rgbStrong[i].x - rgbNone[i].x),
			std::abs(rgbStrong[i].y - rgbNone[i].y),
			std::abs(rgbStrong[i].z - rgbNone[i].z),
		});
	}
	CHECK(rgbMaxDiff <= 1e-6 * maxY);

	sceneStrong.destroy();
	sceneTiny.destroy();
	sceneNone.destroy();
}

TEST_CASE("wavefront dispersion: reflection fresnel follows n(λ)") {
	WavefrontRenderConfig cfg {};
	cfg.sampleCount = test::statistical_samples(2048u);
	cfg.spectral = true;
	WavefrontView const view = dispersion_view(cfg);

	// -- renders: no dispersion, strong dispersion (V = 5)
	MorScene sceneNone = build_dispersive_floor(0.0f, 20.0f);
	MorScene sceneStrong = build_dispersive_floor(4.0f, 20.0f);
	std::vector<PixelXyz> const spectralNone = (
		render_scene_wavefront_xyz(sceneNone, view, cfg)
	);
	std::vector<PixelXyz> const spectralStrong = (
		render_scene_wavefront_xyz(sceneStrong, view, cfg)
	);
	std::vector<bool> const mask = (
		dispersion_floor_mask(view, cfg, spectralNone)
	);
	DispersionSummary const none = summarize_dispersion(spectralNone, mask);
	DispersionSummary const strong = summarize_dispersion(spectralStrong, mask);

	// -- blue has the higher ior, so it reflects more
	f64 const blueShift = (strong.z / strong.x) / (none.z / none.x);
	MESSAGE(
		"strong dispersion z/x shift ", (blueShift - 1.0) * 100.0, "%",
		"  y ", (strong.y / none.y - 1.0) * 100.0, "%"
	);
	CHECK(blueShift > 1.2);
	// V = 5 puts n(λ) above n_d over most of the luminance band, measured
	// +12.5% (2026-11-22)
	CHECK(std::abs(strong.y / none.y - 1.0) < 0.25);

	sceneStrong.destroy();
	sceneNone.destroy();
}
