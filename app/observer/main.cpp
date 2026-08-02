// ponder observer, real-time rasterizer

#include <vkof/vkof.hpp>
#include <srat/camera.hpp>
#include <srat/core-math.hpp>
#include <srat/core-types.hpp>

#include <mor/mor.hpp>
#include <mor/mor-shared.h>

#include <ponder/asset-browser.hpp>
#include <ponder/bluenoise.hpp>
#include <ponder/camera-controller.hpp>
#include <ponder/energy-tables.hpp>
#include <ponder/environment-ibl.hpp>
#include <ponder/environment-tables.hpp>
#include <ponder/material-inspector.hpp>
#include <ponder/vdb.hpp>
#include <ponder/zeltner-tables.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "shaders/scene-visibility-resolve-pc.h"
#include "shaders/shared/global-pc.h"
#include "shaders/shared/scene.h"

#include <algorithm>

// -----------------------------------------------------------------------------
// -- private
// -----------------------------------------------------------------------------

// internal render resolution presets for the imgui dropdown; the scene
// renders at this size regardless of the window size (the finalImage blit
// scales to the swapchain) -- mirrors app/viewer/main.cpp's own list
struct RenderResPreset {
	char const * label;
	u32 w;
	u32 h;
};
constexpr RenderResPreset skRenderResPresets[] = {
	{ "480p", 854u, 480u },
	{ "720p", 1280u, 720u },
	{ "1080p", 1920u, 1080u },
	{ "1440p", 2560u, 1440u },
	{ "2160p", 3840u, 2160u },
	{ "4320p", 7680u, 4320u },
};

// -----------------------------------------------------------------------------
// -- model
// -----------------------------------------------------------------------------

struct ModelInstance {
	mor::Scene scene;
	mor::GpuScene gpuScene;
	u32 modelId;
	// vkof::AccelerationStructureBlas blas; // TODO mega-geom?
	f32v3 boundsMin;
	f32v3 boundsMax;
	std::vector<mor::MeshletAreaInfo> meshletAreas;
};

ModelInstance model_instance_load(char const * const path) {
	mor::Scene const scene = mor::scene_create();
	mor::scene_load_gltf(scene, path);
	f32v3 boundsMin, boundsMax;
	mor::scene_bounds(scene, boundsMin, boundsMax);
	return ModelInstance {
		.scene = scene,
		.gpuScene = mor::scene_gpu_upload(scene),
		.modelId = 0u, // TODO
		.boundsMin = boundsMin,
		.boundsMax = boundsMax,
		.meshletAreas = mor::scene_meshlet_area_info(scene),
	};
}

void model_instance_destroy(ModelInstance & m) {
	mor::scene_gpu_destroy(m.gpuScene);
	mor::scene_destroy(m.scene);
}

// -----------------------------------------------------------------------------
// -- entry point
// -----------------------------------------------------------------------------

int32_t main(int32_t const argc, char const * const * const argv) {
	// TODO use ViewerArgs from viewer
	// TODO below is temp
	// -- initialize application arguments
	char const * const defaultPath = (
		"assets/models-categorized/production-scene/ABeautifulGame.glb"
	);
	char const * const defaultEnvPath = (
		"assets/environments/kloofendal_48d_partly_cloudy_puresky_4k.exr"
	);

	// -- initialize ponder
	vkof::init(/*enableValidation=*/ true);
	vkof::render_extent_set(1280, 720);

	std::filesystem::path const appDir = (
		std::filesystem::canonical(
			std::filesystem::path(__FILE__).parent_path()
		)
	);
	std::filesystem::path const appShaderDir = appDir/"shaders";

	// -- generate tables
	ponder::EnergyTables const energyTables = ponder::energy_tables_create();
	ponder::ZeltnerTables zeltnerTables = ponder::zeltner_tables_create();
	// ponder::BlueNoise const blueNoise = ponder::blue_noise_generate();

	ponder::EnvironmentTables envTables = (
		ponder::environment_tables_create(defaultEnvPath)
	);
	std::string sCurrentEnvMapPath = defaultEnvPath;
	ponder::EnvironmentIbl envIbl = ponder::environment_ibl_create();
	bool envIblBaked = false;

	// debug visualization: one imgui texture id per specular mip + one for
	// irradiance. valid across re-bakes (fnLoadEnvMap rewrites the same
	// images/views in place, never recreates them), so these are created
	// once up front rather than per-frame
	std::vector<ImTextureID> sIblSpecularMipImguiIds;
	for (u32 mip = 0u; mip < ponder::EnvironmentIbl::skSpecularMipCount; ++mip) {
		sIblSpecularMipImguiIds.emplace_back(
			vkof::image_imgui_id({
				.image = envIbl.specularImage,
				.sampler = envIbl.sampler,
				.mipLevel = mip,
			})
		);
	}
	ImTextureID const sIblIrradianceImguiId = vkof::image_imgui_id({
		.image = envIbl.irradianceImage,
		.sampler = envIbl.sampler,
		.mipLevel = 0u,
	});

	// -- pipelines TODO
	char const * const includePaths[] = {
		PONDER_SHADER_DIR,
		MOR_INCLUDE_DIR,
	};
	static constexpr vkof::ImageFormat skVisibilityPipelineColorFormat = (
		vkof::ImageFormat::r32ui
	);
	vkof::Pipeline const skVisibilityDrawPipeline = (
		vkof::pipeline_graphics_create({
			.pathMesh = (appShaderDir/"scene.mesh").string().c_str(),
			.pathFragment = (appShaderDir/"scene.frag").string().c_str(),
			.attachmentColorFormats = (
				srat::slice<vkof::ImageFormat const>(
					&skVisibilityPipelineColorFormat, 1u
				)
			),
			.attachmentDepthStencilFormat = vkof::ImageFormat::d24_unorm_s8_uint,
			.depthTest = vkof::DepthTest::write_on_test_on,
			.cullMode = vkof::CullMode::back,
			.blendMode = vkof::BlendMode::none,
			.includePaths = srat::slice { includePaths, 2u },
		})
	);
	vkof::Pipeline const skVisibilityResolvePipeline = (
		vkof::pipeline_compute_create({
			.pathCompute = (
				(appShaderDir/"scene-visibility-resolve.comp").string().c_str()
			),
			.includePaths = srat::slice { includePaths, 2u },
		})
	);

	// -- render targets
	vkof::TransientImage const visibilityTarget = (
		vkof::transient_image_create({
			.format = vkof::ImageFormat::r32ui,
			.scaleWidth = 1.0f,
			.scaleHeight = 1.0f,
			.mipLevels = 1u,
			.isDoubleBuffered = false,
		})
	);
	vkof::TransientImage const colorTarget = (
		vkof::transient_image_create({
			.format = vkof::ImageFormat::r16g16b16a16_sfloat,
			.scaleWidth = 1.0f,
			.scaleHeight = 1.0f,
			.mipLevels = 1u,
			.isDoubleBuffered = false,
		})
	);
	vkof::TransientImage const depthTarget = (
		vkof::transient_image_create({
			.format = vkof::ImageFormat::d24_unorm_s8_uint,
			.scaleWidth = 1.0f,
			.scaleHeight = 1.0f,
			.mipLevels = 1u,
			.isDoubleBuffered = false,
		})
	);

	// -- internal render resolution
	u32 internalWidth = 1280u;
	u32 internalHeight = 720u;

	// -- settings.json model library, for the asset-browser import panel
	std::filesystem::path const settingsPath = (
		appDir.parent_path().parent_path() / "settings.json"
	);
	std::vector<ponder::ModelEntry> sModelList = ponder::scan_asset_paths(
		settingsPath, { "modelPaths" }, { ".gltf", ".glb" },
		/*warnMissingKey=*/true
	);
	std::vector<ponder::ModelEntry> sEnvMapList = ponder::scan_asset_paths(
		settingsPath, { "environmentMapPaths", "environmentMaps" }, { ".exr" },
		/*warnMissingKey=*/false
	);

	// -- models
	std::vector<ModelInstance> models;
	models.emplace_back(model_instance_load(defaultPath));
	std::string sCurrentModelPath = defaultPath;
	f32 boundsExtent = (
		f32v3_length(models[0].boundsMax - models[0].boundsMin)
	);

	// live-editable copy of the model's materials, rendered from instead of
	// the gltf-sourced original so the material inspector's edits are
	// visible without touching mor's own scene data
	mor::GpuMaterials materialsOverride = (
		mor::scene_gpu_materials_create(models[0].scene)
	);

	// -- models-indirect buffer, per-model geometry/material buffer references
	mor::Buffers modelBufs = mor::scene_gpu_buffers(models[0].gpuScene);
	vkof::Buffer const modelsIndirectBuffer = vkof::buffer_create({
		.byteCount = sizeof(GpuResolveModelIndirect),
		.memory = vkof::BufferMemory::HostWritable,
	});
	auto const fnUploadModelIndirect = [&]() {
		GpuResolveModelIndirect const modelIndirectDesc = {
			.meshlets = modelBufs.meshlets,
			.materials = mor::scene_gpu_materials_va(materialsOverride),
			.uvTransforms = modelBufs.uvTransforms,
			.positions = modelBufs.positions,
			.instances = modelBufs.instances,
			.attributes = modelBufs.attributes,
			.meshletVerts = modelBufs.meshletVerts,
			.meshletTris = modelBufs.meshletTris,
			.modelMatrix = f32m44_identity(),
		};
		vkof::buffer_upload({
			.buffer = modelsIndirectBuffer,
			.byteOffset = 0u,
			.data = srat::slice_as_bytes(modelIndirectDesc),
		});
	};
	fnUploadModelIndirect();
	u64 const modelsIndirectVa = (
		vkof::buffer_virtual_address(modelsIndirectBuffer)
	);

	// -- camera
	ponder::CameraController camera = ponder::camera_controller_create({
		.target = (models[0].boundsMin + models[0].boundsMax) * 0.5f,
		.distance = boundsExtent * 1.5f,
		.azimuth = 0.0f,
		.elevation = 0.3f,
		.fovY = 1.0f,
		.aspect = (f32)internalWidth / (f32)internalHeight,
		.near = boundsExtent * 0.001f,
		.far = boundsExtent * 100.0f,
	});

	// -- tonemap
	f32 sExposure = 1.0f;

	// -- material inspector probe: right-click sets sProbePixel for the
	// next dispatch; the resolve pass writes the hit's (modelDrawIndex,
	// materialIndex) into probeResultBuffer, a plain host-readable buffer
	i32v2 sProbePixel { -1, -1 };
	i32 sPickedMaterialIndex = -1;
	bool sMaterialPopupPending = false;
	vkof::Buffer const probeResultBuffer = vkof::buffer_create({
		.byteCount = sizeof(GpuProbeResult),
		.memory = vkof::BufferMemory::HostWritable,
	});
	auto const fnResetProbeResult = [&]() {
		GpuProbeResult const sentinel { -1, -1 };
		vkof::buffer_upload({
			.buffer = probeResultBuffer,
			.byteOffset = 0u,
			.data = srat::slice_as_bytes(sentinel),
		});
	};
	fnResetProbeResult();

	// -- swaps the active model wholesale: waits for the gpu to go idle
	// (never call this from inside an open ImGui window), tears down the
	// old model + materialsOverride, loads path, and re-frames the camera
	auto const fnLoadModel = [&](char const * const path) {
		vkof::device_wait_idle();
		mor::scene_gpu_materials_destroy(materialsOverride);
		model_instance_destroy(models[0]);

		models[0] = model_instance_load(path);
		sCurrentModelPath = path;
		materialsOverride = mor::scene_gpu_materials_create(models[0].scene);
		modelBufs = mor::scene_gpu_buffers(models[0].gpuScene);
		fnUploadModelIndirect();

		boundsExtent = (
			f32v3_length(models[0].boundsMax - models[0].boundsMin)
		);
		camera.orbit.target = (
			(models[0].boundsMin + models[0].boundsMax) * 0.5f
		);
		camera.orbit.distance = boundsExtent * 1.5f;
		camera.orbit.near = boundsExtent * 0.001f;
		camera.orbit.far = boundsExtent * 100.0f;
		ponder::camera_controller_sync_fp_from_orbit(camera);

		sPickedMaterialIndex = -1;
	};

	// swaps the active environment map: waits for the gpu to go idle (never
	// call from inside an open ImGui window), reloads envTables, and marks
	// the ibl bake to re-run on the next frame (writes into the existing
	// envIbl images, no need to recreate them)
	auto const fnLoadEnvMap = [&](std::string const & path) {
		vkof::device_wait_idle();
		if (envTables.width != 0u) {
			ponder::environment_tables_destroy(envTables);
			envTables = {};
		}
		sCurrentEnvMapPath.clear();
		envTables = ponder::environment_tables_create(path);
		if (envTables.width == 0u) {
			return;
		}
		sCurrentEnvMapPath = path;
		envIblBaked = false;
	};

	// -- render loop
	auto const fnRenderFrame = [&](
	) {
		std::vector<vkof::RenderNode> nodes;

		if (!envIblBaked && envTables.width != 0u) {
			ponder::environment_ibl_bake(envIbl, envTables, nodes);
			envIblBaked = true;
		}

		GpuGlobalPc globalPc = {
			.time = 0.0f,
			.cameraPos = (
				camera.isFirstPerson
				? camera.firstPerson.position
				: srat::camera_orbit_eye(camera.orbit)
			),
			.exposure = sExposure,
			.pad0 = 0.0f,
			.viewProj = (
				camera.isFirstPerson
				? (
					srat::camera_fp_proj(camera.firstPerson)
					* srat::camera_fp_view(camera.firstPerson)
				)
				: (
					srat::camera_orbit_proj(camera.orbit)
					* srat::camera_orbit_view(camera.orbit)
				)
			),
			// .extended = vkof::buffer_virtual_address(...),
			.models = modelsIndirectVa,
			.renderWidth = internalWidth,
			.renderHeight = internalHeight,
			.pad1 = 0u,
			.pad2 = 0u,
			.pad3 = 0u,
		};

		u32 const visibilityHandle = (
			vkof::transient_image_storage_handle({
				.image = visibilityTarget,
				.mipLevel = 0u,
			})
		);
		u32 const colorHandle = (
			vkof::transient_image_storage_handle({
				.image = colorTarget,
				.mipLevel = 0u,
			})
		);
		u32 const depthHandle = (
			vkof::transient_image_storage_handle({
				.image = depthTarget,
				.mipLevel = 0u,
			})
		);

		// -- render visibility pass
		vkof::RenderNode const visibilityDrawNode = (
			vkof::render_node_create({
				.queue = vkof::CommandQueue::graphics,
			})
		);
		static constexpr f32 skDrawVisibilityClearColor[4] = {
			0.0f, 0.0f, 0.0f, 0.0f
		};
		static constexpr f32 skDrawVisibilityClearDepth = 1.0f;
		vkof::render_node_attachment_color({
			.node = visibilityDrawNode,
			.image = visibilityTarget,
			.loadOp = vkof::RenderNodeLoadOp::clear,
			.mipLevel = 0u,
			.colorIndex = 0u,
			.clearColor = srat::slice<f32 const>(skDrawVisibilityClearColor, 4),
		});
		vkof::render_node_attachment_depth({
			.node = visibilityDrawNode,
			.image = depthTarget,
			.loadOp = vkof::RenderNodeLoadOp::clear,
			.mipLevel = 0u,
			.clearDepth = srat::slice<f32 const>(&skDrawVisibilityClearDepth, 1u),
		});
		vkof::render_node_callback({
			.node = visibilityDrawNode,
			.callback = [&](vkof::CommandBuffer const & cmd) {
				GpuSceneDrawPc const drawPc = {
					.modelId = models[0].modelId,
					.meshlets = modelBufs.meshlets,
					.positions = modelBufs.positions,
					.instances = modelBufs.instances,
					.meshletTris = modelBufs.meshletTris,
					.meshletVerts = modelBufs.meshletVerts,
					.modelMatrix = f32m44_identity(),
				};
				vkof::cmd_draw_pushconst(vkof::CmdDrawPushconst {
					.cmd = cmd,
					.pipeline = skVisibilityDrawPipeline,
					.push = drawPc,
					.vertexCount = mor::scene_gpu_meshlet_count(models[0].gpuScene),
					.instanceCount = 1u,
				});
			},
		});
		nodes.emplace_back(visibilityDrawNode);

		// -- visibility resolve pass
		vkof::RenderNode const visibilityResolveNode = (
			vkof::render_node_create({
				.queue = vkof::CommandQueue::graphics,
			})
		);
		vkof::render_node_add_image({
			.node = visibilityResolveNode,
			.image = visibilityTarget,
			.access = vkof::RenderNodeAccess::read,
		});
		vkof::render_node_add_image({
			.node = visibilityResolveNode,
			.image = depthTarget,
			.access = vkof::RenderNodeAccess::read,
		});
		vkof::render_node_add_image({
			.node = visibilityResolveNode,
			.image = colorTarget,
			.access = vkof::RenderNodeAccess::write,
		});
		vkof::render_node_add_persistent_image({
			.node = visibilityResolveNode,
			.image = envIbl.specularImage,
			.access = vkof::RenderNodeAccess::read,
		});
		vkof::render_node_add_persistent_image({
			.node = visibilityResolveNode,
			.image = envIbl.irradianceImage,
			.access = vkof::RenderNodeAccess::read,
		});
		static u32 frameIndex = 0u; // TODO make this global
		++frameIndex;
		u32 const writeIdx = frameIndex % 2u;
		vkof::render_node_callback({
			.node = visibilityResolveNode,
			.callback = [&](vkof::CommandBuffer const & cmd) {
				GpuResolvePc const resolvePc = {
					.visibilityImageHandle = visibilityHandle,
					.outputImageHandle = colorHandle,
					.frameIndex = frameIndex,
					.kullaContyEnergyHandle = energyTables.kullaContyEnergyHandle,
					.zeltnerLtcParamHandle = zeltnerTables.zeltnerLtcParamHandle,
					.probePixel = sProbePixel,
					.probeResultVa = vkof::buffer_virtual_address(probeResultBuffer),
					.envRadianceHandle = (
						envTables.width != 0u ? envTables.radianceHandle : 0u
					),
					.envIblSpecularHandle = envIbl.specularHandle,
					.envIblIrradianceHandle = envIbl.irradianceHandle,
					.envIblSpecularMaxLod = (
						(f32)(ponder::EnvironmentIbl::skSpecularMipCount - 1u)
					),
				};
				vkof::cmd_dispatch_pushconst(vkof::CmdDispatchPushconst {
					.cmd = cmd,
					.pipeline = skVisibilityResolvePipeline,
					.push = resolvePc,
					.threadgroupSize = u32v3 { 16u, 16u, 1u },
					.invocationCount = u32v3 { internalWidth, internalHeight, 1u },
				});
			},
		});
		nodes.emplace_back(visibilityResolveNode);

		// -- render graph execute
		vkof::render_graph_execute({
			.nodes = (
				srat::slice<vkof::RenderNode const>(nodes.data(), nodes.size())
			),
			.rootPushconstant = srat::slice_as_bytes(globalPc),
			.finalImage = colorTarget,
			.debugDrawViewProj = globalPc.viewProj,
			.debugDrawDepth = depthTarget,
		});

		// TODO this should be implicit by render graph execute i guess
		vkof::render_node_destroy(visibilityDrawNode);
		vkof::render_node_destroy(visibilityResolveNode);
	};

	// -- observer loop
	GLFWwindow * const window = vkof::window();
	ponder::camera_controller_install_scroll_callback(window);
	bool rightMouseWasDown = false;
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
			glfwSetWindowShouldClose(window, GLFW_TRUE);
		}

		vkof::imgui_begin();
		ponder::camera_controller_update(
			camera, window, boundsExtent, ImGui::GetIO().DeltaTime
		);

		// right-click probes the pixel under the cursor for its material;
		// mapped from window coordinates into the internal render resolution
		bool const mouseFree = !ImGui::GetIO().WantCaptureMouse;
		bool const rightMouseDown = (
			glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS
		);
		if (mouseFree && rightMouseDown && !rightMouseWasDown) {
			f64 curMouseX, curMouseY;
			glfwGetCursorPos(window, &curMouseX, &curMouseY);
			i32 windowW, windowH;
			glfwGetWindowSize(window, &windowW, &windowH);
			if (windowW > 0 && windowH > 0) {
				sProbePixel = {
					std::clamp(
						(i32)(curMouseX * (f64)internalWidth / (f64)windowW),
						0, (i32)internalWidth - 1
					),
					std::clamp(
						(i32)(curMouseY * (f64)internalHeight / (f64)windowH),
						0, (i32)internalHeight - 1
					),
				};
				fnResetProbeResult();
			}
		}
		rightMouseWasDown = rightMouseDown;

		// OpenPopup/BeginPopup must share the same id-stack context, so the
		// pending flag from last frame's probe readback (below) is applied
		// here, outside the "observer" window's own id stack
		bool const openMaterialPopup = sMaterialPopupPending;
		sMaterialPopupPending = false;

		// deferred outside the window scope: fnLoadModel/fnLoadEnvMap wait on
		// the gpu and tear down gpu resources, which must never happen while
		// an ImGui::Begin() window is still open
		std::string pendingLoadPath;
		std::string pendingEnvMapPath;
		if (ImGui::Begin("observer")) {
			ImGui::Text("model: %s", sCurrentModelPath.c_str());
			std::string const picked = ponder::asset_list_draw(
				sModelList, sCurrentModelPath, "modelList", 200.0f
			);
			if (!picked.empty()) {
				pendingLoadPath = picked;
			}
		}
		ImGui::End();
		if (ImGui::Begin("environment")) {
			ImGui::Text("env map: %s", sCurrentEnvMapPath.c_str());
			std::string const pickedEnvMap = ponder::asset_list_draw(
				sEnvMapList, sCurrentEnvMapPath, "envMapList", 200.0f
			);
			if (!pickedEnvMap.empty()) {
				pendingEnvMapPath = pickedEnvMap;
			}
		}
		ImGui::End();
		if (ImGui::Begin("render")) {
			// -1 when the current internal resolution matches no preset
			i32 resIdx = -1;
			for (i32 i = 0; i < (i32)IM_ARRAYSIZE(skRenderResPresets); ++i) {
				if (
					skRenderResPresets[i].w == internalWidth
					&& skRenderResPresets[i].h == internalHeight
				) {
					resIdx = i;
					break;
				}
			}
			i32 newIdx = resIdx;
			ImGui::Combo(
				"resolution",
				&newIdx,
				[](void * data, int idx) -> char const * {
					return (
						static_cast<RenderResPreset const *>(data)[idx].label
					);
				},
				(void *)skRenderResPresets,
				(i32)IM_ARRAYSIZE(skRenderResPresets)
			);
			if (newIdx != resIdx && newIdx >= 0) {
				internalWidth = skRenderResPresets[newIdx].w;
				internalHeight = skRenderResPresets[newIdx].h;
				// resizes every transient image (visibilityTarget,
				// colorTarget, depthTarget) in place; observer has no
				// persistent per-resolution images to recreate by hand
				vkof::render_extent_set(internalWidth, internalHeight);
				camera.orbit.aspect = (
					(f32)internalWidth / (f32)internalHeight
				);
				camera.firstPerson.aspect = camera.orbit.aspect;
			}
		}
		ImGui::End();
		if (ImGui::Begin("tonemap")) {
			ImGui::SliderFloat("exposure", &sExposure, 0.01f, 8.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
		}
		ImGui::End();
		if (ImGui::Begin("ibl debug")) {
			static constexpr f32 skPreviewWidth = 256.0f;
			ImGui::Text("irradiance (%ux%u)",
				ponder::EnvironmentIbl::skIrradianceWidth,
				ponder::EnvironmentIbl::skIrradianceHeight
			);
			ImGui::Image(
				sIblIrradianceImguiId,
				ImVec2(
					skPreviewWidth,
					skPreviewWidth
					* (f32)ponder::EnvironmentIbl::skIrradianceHeight
					/ (f32)ponder::EnvironmentIbl::skIrradianceWidth
				)
			);
			ImGui::SeparatorText("specular mips");
			for (
				u32 mip = 0u;
				mip < ponder::EnvironmentIbl::skSpecularMipCount;
				++mip
			) {
				u32 const mipW = (
					std::max(1u, ponder::EnvironmentIbl::skSpecularBaseWidth >> mip)
				);
				u32 const mipH = (
					std::max(1u, ponder::EnvironmentIbl::skSpecularBaseHeight >> mip)
				);
				f32 const roughness = (
					(f32)mip / (f32)(ponder::EnvironmentIbl::skSpecularMipCount - 1u)
				);
				ImGui::Text(
					"mip %u: roughness %.2f (%ux%u)", mip, roughness, mipW, mipH
				);
				ImGui::Image(
					sIblSpecularMipImguiIds[mip],
					ImVec2(skPreviewWidth, skPreviewWidth * (f32)mipH / (f32)mipW)
				);
			}
		}
		ImGui::End();
		if (openMaterialPopup) {
			ImGui::OpenPopup("material inspector");
		}
		if (ImGui::BeginPopup("material inspector")) {
			bool const matChanged = ponder::material_inspector_draw(
				models[0].scene, materialsOverride, sPickedMaterialIndex
			);
			if (matChanged) {
				mor::scene_gpu_materials_upload(materialsOverride);
			}
			ImGui::EndPopup();
		}
		if (!pendingLoadPath.empty()) {
			fnLoadModel(pendingLoadPath.c_str());
		}
		if (!pendingEnvMapPath.empty()) {
			fnLoadEnvMap(pendingEnvMapPath);
		}

		fnRenderFrame();
		sProbePixel = { -1, -1 };

		// the gpu is pipelined deeply enough that this frame's probe-result
		// write can surface well after render_graph_execute returns; read
		// back right after fnRenderFrame instead of at the top of next
		// frame's imgui block, which gives it the largest possible window
		// to have landed
		GpuProbeResult const * const probeResult = (
			reinterpret_cast<GpuProbeResult const *>(
				vkof::buffer_host_address(probeResultBuffer).ptr()
			)
		);
		if (probeResult->modelDrawIndex >= 0) {
			sPickedMaterialIndex = probeResult->materialIndex;
			sMaterialPopupPending = true;
			fnResetProbeResult();
		}
	}

	vkof::device_wait_idle();
	ponder::energy_tables_destroy(energyTables);
	ponder::zeltner_tables_destroy(zeltnerTables);
	if (envTables.width != 0u) {
		ponder::environment_tables_destroy(envTables);
	}
	for (ImTextureID const id : sIblSpecularMipImguiIds) {
		vkof::image_imgui_id_destroy(id);
	}
	vkof::image_imgui_id_destroy(sIblIrradianceImguiId);
	ponder::environment_ibl_destroy(envIbl);
	mor::scene_gpu_materials_destroy(materialsOverride);
	model_instance_destroy(models[0]);
	// TODO destroy pipelines, targets, etc
	vkof::buffer_destroy(modelsIndirectBuffer);
	vkof::buffer_destroy(probeResultBuffer);
	vkof::pipeline_destroy(skVisibilityDrawPipeline);
	vkof::pipeline_destroy(skVisibilityResolvePipeline);

	vkof::shutdown();
}
