#include "scene-state.hpp"

#include <ponder/subsurface-presets.hpp>

#include <algorithm>
#include <cfloat>
#include <cstdio>

// -----------------------------------------------------------------------------
// -- file view scene state
// -----------------------------------------------------------------------------

// glTF has no subsurface extension, so every imported material starts with
// material_load_default()'s (lib/mor/src/mor.cpp) flat subsurfaceColor/
// Radius -- not tied to this specific asset's own scale or color at all.
// seeds a scale/color-appropriate starting point for every material that
// hasn't been given real subsurface data (subsurfaceWeight still at its
// untouched 0.0 default), WITHOUT turning subsurfaceWeight on: only affects
// what a material looks like once the user opts it into subsurface later
// (material-inspector.cpp), never changes how it renders today
static void scene_state_seed_subsurface_defaults(
	mor::GpuMaterials const & materialsOverride, f32v3 const boundsMin, f32v3 const boundsMax
) {
	// generic fallback fraction, not tied to any one named preset (see
	// subsurface-presets.hpp) -- a modest middle-of-the-road scale so a
	// freshly-enabled material isn't absurdly over/under-scattering before
	// any tuning happens
	f32 constexpr skDefaultRadiusFraction = 0.03f;
	u32 const count = mor::scene_gpu_materials_count(materialsOverride);
	bool anyChanged = false;
	for (u32 i = 0u; i < count; ++i) {
		GpuMorMaterial mat = mor::scene_gpu_materials_get(materialsOverride, i);
		if (mat.subsurfaceWeight.r > 0.0f || mat.subsurfaceWeight.texture != 0u) {
			// already has real data (a future gltf extension, or an
			// already-edited scene reloaded) -- leave it alone
			continue;
		}
		// whole-struct copy, not just .rgb: carries baseColor's texture +
		// uvTransform along too, so a textured face/skin material seeds
		// real per-pixel variation instead of just its flat factor (which
		// for a textured material is often close to white/1, since the
		// texture supplies the actual color)
		mat.subsurfaceColor = mat.baseColor;
		mat.subsurfaceRadius.r = (
			ponder::subsurfaceRadiusFromExtent(
				boundsMin, boundsMax, skDefaultRadiusFraction
			)
		);
		mor::scene_material_override_scalars(materialsOverride, i, mat);
		anyChanged = true;
	}
	if (anyChanged) {
		mor::scene_gpu_materials_upload(materialsOverride);
	}
}

void scene_state_rebuild_tlas(SceneState & s, f32m44 const & transform) {
	bool const hasInstance = s.blas.id != 0u;
	vkof::TlasInstance const instance = {
		.blas = s.blas,
		.transform = transform,
		.instanceCustomIndex = 0u,
		.rayMask = 0xFFu,
	};
	vkof::RenderNode const tlasNode = (
		vkof::render_node_create({ .queue = vkof::CommandQueue::compute })
	);
	vkof::render_node_callback({
		.node = tlasNode,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			vkof::tlas_build(
				cmd,
				s.tlas,
				hasInstance
					? srat::slice<vkof::TlasInstance const>(&instance, 1u)
					: srat::slice<vkof::TlasInstance const>(nullptr, 0u)
			);
		},
	});
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(&tlasNode, 1u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(tlasNode);
}

SceneState scene_state_create(char const * const path) {
	SceneState s;
	if (path == nullptr) {
		s.boundsCenter = f32v3 { 0.0f, 0.0f, 0.0f };
		s.boundsExtent = 1.0f;
		s.tlas = vkof::tlas_create({ .maxInstances = 1u });
		scene_state_rebuild_tlas(s, f32m44_identity());
		vkof::acceleration_structure_set_tlas(s.tlas);
		s.debugPcBuffer = vkof::buffer_create({
			.byteCount = sizeof(GpuGlobalExtended),
			.memory = vkof::BufferMemory::HostWritable,
		});
		s.modelsIndirectBuffer = vkof::buffer_create({
			.byteCount = sizeof(GpuResolveModelIndirect),
			.memory = vkof::BufferMemory::HostWritable,
		});
		return s;
	}
	s.scene = mor::scene_create();
	mor::scene_load_gltf(s.scene, path);
	s.gpuScene = mor::scene_gpu_upload(s.scene);
	s.bufs = mor::scene_gpu_buffers(s.gpuScene);
	s.materialsOverride = mor::scene_gpu_materials_create(s.scene);

	f32v3 boundsMin, boundsMax;
	mor::scene_bounds(s.scene, boundsMin, boundsMax);
	s.boundsCenter = (boundsMin + boundsMax) * 0.5f;
	s.boundsExtent = std::max(
		{
			boundsMax.x - boundsMin.x,
			boundsMax.y - boundsMin.y,
			boundsMax.z - boundsMin.z,
			0.01f,
		}
	);
	scene_state_seed_subsurface_defaults(s.materialsOverride, boundsMin, boundsMax);

	s.blas = vkof::blas_create({
		.positionVa = s.bufs.positions,
		.vertexCount = s.bufs.vertexCount,
		.indexVa = s.bufs.flatIndices,
		.triangleCount = s.bufs.triangleCount,
		.isOpaque = mor::scene_is_fully_opaque(s.scene),
	});
	if (s.blas.id == 0u) {
		printf("failed to build blas for '%s'\n", path);
		mor::scene_gpu_destroy(s.gpuScene);
		mor::scene_destroy(s.scene);
		return s;
	}
	s.tlas = vkof::tlas_create({ .maxInstances = 1u });
	scene_state_rebuild_tlas(s, f32m44_identity());
	vkof::acceleration_structure_set_tlas(s.tlas);

	s.debugPcBuffer = vkof::buffer_create({
		.byteCount = sizeof(GpuGlobalExtended),
		.memory = vkof::BufferMemory::HostWritable,
	});
	s.modelsIndirectBuffer = vkof::buffer_create({
		.byteCount = sizeof(GpuResolveModelIndirect),
		.memory = vkof::BufferMemory::HostWritable,
	});
	return s;
}

void scene_state_destroy(SceneState & s) {
	vkof::device_wait_idle();
	if (s.modelsIndirectBuffer.id != 0u) {
		vkof::buffer_destroy(s.modelsIndirectBuffer);
	}
	if (s.debugPcBuffer.id != 0u) {
		vkof::buffer_destroy(s.debugPcBuffer);
	}
	if (s.tlas.id != 0u) {
		vkof::tlas_destroy(s.tlas);
	}
	if (s.blas.id != 0u) {
		vkof::blas_destroy(s.blas);
	}
	if (s.materialsOverride.id != 0u) {
		mor::scene_gpu_materials_destroy(s.materialsOverride);
	}
	// the empty (--vdb-only) scene never creates these; mor's destroys
	// dereference the handle unguarded
	if (s.gpuScene.id != 0u) {
		mor::scene_gpu_destroy(s.gpuScene);
	}
	if (s.scene.id != 0u) {
		mor::scene_destroy(s.scene);
	}
	s = {};
}

void stage_render_rebuild_blases(
	StageRenderState & r, stage::Stage const & stage
) {
	vkof::device_wait_idle();
	for (vkof::AccelerationStructureBlas & b : r.blases) {
		if (b.id != 0u) { vkof::blas_destroy(b); }
	}
	r.blases.clear();
	r.blases.reserve(stage.gltfMeshes.size());
	for (stage::GltfMesh const & mesh : stage.gltfMeshes) {
		if (mesh.gpuScene.id == 0u) {
			r.blases.emplace_back(vkof::AccelerationStructureBlas {});
			continue;
		}
		mor::Buffers const bufs = mor::scene_gpu_buffers(mesh.gpuScene);
		r.blases.emplace_back(vkof::blas_create({
			.positionVa = bufs.positions,
			.vertexCount = bufs.vertexCount,
			.indexVa = bufs.flatIndices,
			.triangleCount = bufs.triangleCount,
			.isOpaque = mor::scene_is_fully_opaque(mesh.scene),
		}));
	}
	if (r.tlas.id != 0u) {
		vkof::tlas_destroy(r.tlas);
	}
	r.tlas = vkof::tlas_create({
		.maxInstances = (u32)std::max<size_t>(stage.gltfInstances.size(), 1u),
	});
	if (r.debugPcBuffer.id == 0u) {
		r.debugPcBuffer = vkof::buffer_create({
			.byteCount = sizeof(GpuGlobalExtended),
			.memory = vkof::BufferMemory::HostWritable,
		});
	}
	vkof::acceleration_structure_set_tlas(r.tlas);
}

// -----------------------------------------------------------------------------
// -- stage render state
// -----------------------------------------------------------------------------

void stage_render_rebuild_tlas(StageRenderState & r, stage::Stage const & stage) {
	std::vector<vkof::TlasInstance> tlasInstances;
	std::vector<GpuResolveModelIndirect> models;
	r.modelDrawIndexToInstance.clear();
	f32v3 boundsMin { FLT_MAX, FLT_MAX, FLT_MAX };
	f32v3 boundsMax { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	bool anyBounds = false;
	for (size_t i = 0u; i < stage.gltfInstances.size(); ++i) {
		stage::GltfInstance const & inst = stage.gltfInstances[i];
		if (!inst.visible || inst.modelIndex >= stage.gltfModels.size()) {
			continue;
		}
		stage::GltfModel const & model = stage.gltfModels[inst.modelIndex];
		if (
			model.meshIndex >= r.blases.size()
			|| r.blases[model.meshIndex].id == 0u
		) {
			continue;
		}
		stage::GltfMesh const & mesh = stage.gltfMeshes[model.meshIndex];
		f32m44 const transform = stage::transform_to_m44(inst.transform);
		tlasInstances.emplace_back(vkof::TlasInstance {
			.blas = r.blases[model.meshIndex],
			.transform = transform,
			.instanceCustomIndex = (u32)models.size(),
			.rayMask = 0xFFu,
		});
		r.modelDrawIndexToInstance.emplace_back(i);
		mor::Buffers const bufs = mor::scene_gpu_buffers(mesh.gpuScene);
		u64 const materialsVa = (
			model.materials.id != 0u
				? mor::scene_gpu_materials_va(model.materials)
				: bufs.materials
		);
		models.emplace_back(GpuResolveModelIndirect {
			.meshlets = bufs.meshlets,
			.materials = materialsVa,
			.uvTransforms = bufs.uvTransforms,
			.positions = bufs.positions,
			.instances = bufs.instances,
			.attributes = bufs.attributes,
			.meshletVerts = bufs.meshletVerts,
			.meshletTris = bufs.meshletTris,
			.flatIndices = bufs.flatIndices,
			.flatMeshlets = bufs.flatMeshlets,
			.modelMatrix = transform,
		});
		f32v3 instWorldMin, instWorldMax;
		if (
			stage::gltf_instance_world_bounds(stage, inst, instWorldMin, instWorldMax)
		) {
			boundsMin.x = std::min(boundsMin.x, instWorldMin.x);
			boundsMin.y = std::min(boundsMin.y, instWorldMin.y);
			boundsMin.z = std::min(boundsMin.z, instWorldMin.z);
			boundsMax.x = std::max(boundsMax.x, instWorldMax.x);
			boundsMax.y = std::max(boundsMax.y, instWorldMax.y);
			boundsMax.z = std::max(boundsMax.z, instWorldMax.z);
			anyBounds = true;
		}
	}
	for (stage::VdbInstance const & inst : stage.vdbInstances) {
		if (!inst.visible) { continue; }
		f32v3 instWorldMin, instWorldMax;
		if (!stage::vdb_instance_world_bounds(inst, instWorldMin, instWorldMax)) {
			continue;
		}
		boundsMin.x = std::min(boundsMin.x, instWorldMin.x);
		boundsMin.y = std::min(boundsMin.y, instWorldMin.y);
		boundsMin.z = std::min(boundsMin.z, instWorldMin.z);
		boundsMax.x = std::max(boundsMax.x, instWorldMax.x);
		boundsMax.y = std::max(boundsMax.y, instWorldMax.y);
		boundsMax.z = std::max(boundsMax.z, instWorldMax.z);
		anyBounds = true;
	}
	vkof::RenderNode const tlasNode = (
		vkof::render_node_create({ .queue = vkof::CommandQueue::compute })
	);
	vkof::render_node_callback({
		.node = tlasNode,
		.callback = [&](vkof::CommandBuffer const & cmd) {
			vkof::tlas_build(
				cmd,
				r.tlas,
				srat::slice<vkof::TlasInstance const>(
					tlasInstances.data(), tlasInstances.size()
				)
			);
		},
	});
	vkof::render_graph_execute({
		.nodes = srat::slice<vkof::RenderNode const>(&tlasNode, 1u),
		.rootPushconstant = srat::slice<u8 const>(nullptr, 0u),
	});
	vkof::render_node_destroy(tlasNode);

	if (models.size() > r.modelsIndirectCapacity) {
		if (r.modelsIndirectBuffer.id != 0u) {
			vkof::device_wait_idle();
			vkof::buffer_destroy(r.modelsIndirectBuffer);
		}
		r.modelsIndirectCapacity = (u32)models.size();
		r.modelsIndirectBuffer = vkof::buffer_create({
			.byteCount = (
				sizeof(GpuResolveModelIndirect)
				* std::max<u32>(r.modelsIndirectCapacity, 1u)
			),
			.memory = vkof::BufferMemory::HostWritable,
		});
	}
	if (!models.empty()) {
		vkof::buffer_upload({
			.buffer = r.modelsIndirectBuffer,
			.byteOffset = 0u,
			.data = srat::slice<u8 const>(
				reinterpret_cast<u8 const *>(models.data()),
				models.size() * sizeof(GpuResolveModelIndirect)
			),
		});
	}

	if (anyBounds) {
		r.boundsCenter = (boundsMin + boundsMax) * 0.5f;
		r.boundsExtent = std::max(
			{
				boundsMax.x - boundsMin.x,
				boundsMax.y - boundsMin.y,
				boundsMax.z - boundsMin.z,
				0.01f,
			}
		);
	} else {
		r.boundsCenter = { 0.0f, 0.0f, 0.0f };
		r.boundsExtent = 1.0f;
	}
}

void stage_render_destroy(StageRenderState & r) {
	vkof::device_wait_idle();
	for (vkof::AccelerationStructureBlas & b : r.blases) {
		if (b.id != 0u) { vkof::blas_destroy(b); }
	}
	r.blases.clear();
	if (r.tlas.id != 0u) {
		vkof::tlas_destroy(r.tlas);
	}
	if (r.modelsIndirectBuffer.id != 0u) {
		vkof::buffer_destroy(r.modelsIndirectBuffer);
	}
	if (r.debugPcBuffer.id != 0u) {
		vkof::buffer_destroy(r.debugPcBuffer);
	}
	r = {};
}
