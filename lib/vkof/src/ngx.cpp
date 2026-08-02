#include <vkof/ngx.hpp>

#include "ngx-internal.hpp"

// helpers_vk.h defines what helpers_dlssd_vk.h uses
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_helpers_dlssd_vk.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

// any stable uuid; ngx keys its per-application data on it
static constexpr char const * skProjectId = "6f1d3c2e-8a4b-4c5d-9e7f-0a1b2c3d4e5f";

struct NgxState {
	bool initialized {};
	NVSDK_NGX_Parameter * capabilities {};
	bool rrAvailable {};
	std::string rrUnavailableReason { "ngx not initialized" };
	// the ray reconstruction feature and what it was created with
	NVSDK_NGX_Handle * rr {};
	NVSDK_NGX_Parameter * rrParameters {};
	u32v2 rrRenderSize {};
	u32v2 rrOutputSize {};
	vkof::NgxRrQuality rrQuality {};
	vkof::NgxRrPreset rrPreset {};
	// the last render size query
	vkof::NgxRrQuality sizeQuality {};
	u32v2 sizeOutput {};
	u32v2 sizeRender {};
	std::wstring exeDir {};
};

NgxState sNgx {};

void NVSDK_CONV log_callback(
	char const * const message,
	NVSDK_NGX_Logging_Level const,
	NVSDK_NGX_Feature const
) {
	printf("[ngx] %s", message);
}

std::string result_string(NVSDK_NGX_Result const result) {
	char buffer[32];
	snprintf(buffer, sizeof(buffer), "0x%08x", (u32)result);
	return buffer;
}

NVSDK_NGX_Resource_VK resource(
	vkof::Image const & image,
	bool const readWrite
) {
	vkof::internal::NativeImage native {};
	if (!vkof::internal::native_image(image, native)) {
		return NVSDK_NGX_Resource_VK {};
	}
	auto const range = VkImageSubresourceRange {
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.baseMipLevel = 0u,
		.levelCount = 1u,
		.baseArrayLayer = 0u,
		.layerCount = 1u,
	};
	return (
		NVSDK_NGX_Create_ImageView_Resource_VK(
			native.view, native.image, range, native.format,
			native.width, native.height, readWrite
		)
	);
}

} // namespace

void vkof::ngx::required_extensions(
	std::vector<char const *> & instanceExtensions,
	std::vector<char const *> & deviceExtensions
) {
	unsigned int instanceCount = 0u;
	char const ** instanceNames = nullptr;
	unsigned int deviceCount = 0u;
	char const ** deviceNames = nullptr;
	NVSDK_NGX_Result const result = (
		NVSDK_NGX_VULKAN_RequiredExtensions(
			&instanceCount, &instanceNames, &deviceCount, &deviceNames
		)
	);
	if (NVSDK_NGX_FAILED(result)) {
		printf(
			"[ngx] required extensions query failed: %s\n",
			result_string(result).c_str()
		);
		return;
	}
	for (u32 i = 0u; i < instanceCount; ++i) {
		instanceExtensions.emplace_back(instanceNames[i]);
	}
	for (u32 i = 0u; i < deviceCount; ++i) {
		deviceExtensions.emplace_back(deviceNames[i]);
	}
}

void vkof::ngx::init() {
	std::error_code ec;
	std::filesystem::path const exeDir = (
		std::filesystem::read_symlink("/proc/self/exe", ec).parent_path()
	);
	sNgx.exeDir = exeDir.wstring();
	wchar_t const * const paths[] = { sNgx.exeDir.c_str() };
	bool const verbose = std::getenv("PONDER_NGX_LOG") != nullptr;
	auto commonInfo = NVSDK_NGX_FeatureCommonInfo {};
	commonInfo.PathListInfo = NVSDK_NGX_PathListInfo {
		.Path = paths,
		.Length = 1u,
	};
	commonInfo.LoggingInfo = NVSDK_NGX_LoggingInfo {
		.LoggingCallback = verbose ? &log_callback : nullptr,
		.MinimumLoggingLevel = (
			verbose ? NVSDK_NGX_LOGGING_LEVEL_VERBOSE : NVSDK_NGX_LOGGING_LEVEL_OFF
		),
		.DisableOtherLoggingSinks = verbose,
	};
	NVSDK_NGX_Result const initResult = (
		NVSDK_NGX_VULKAN_Init_with_ProjectID(
			skProjectId,
			NVSDK_NGX_ENGINE_TYPE_CUSTOM,
			"0",
			sNgx.exeDir.c_str(),
			internal::native_instance(),
			internal::native_physical_device(),
			internal::native_device(),
			vkGetInstanceProcAddr,
			vkGetDeviceProcAddr,
			&commonInfo
		)
	);
	if (NVSDK_NGX_FAILED(initResult)) {
		sNgx.rrUnavailableReason = (
			"ngx init failed (" + result_string(initResult) + ")"
		);
		printf("[ngx] %s\n", sNgx.rrUnavailableReason.c_str());
		return;
	}
	sNgx.initialized = true;

	// -- ray reconstruction availability
	NVSDK_NGX_Result const capResult = (
		NVSDK_NGX_VULKAN_GetCapabilityParameters(&sNgx.capabilities)
	);
	if (NVSDK_NGX_FAILED(capResult)) {
		sNgx.rrUnavailableReason = (
			"ngx capability query failed (" + result_string(capResult) + ")"
		);
		printf("[ngx] %s\n", sNgx.rrUnavailableReason.c_str());
		return;
	}
	int needsUpdatedDriver = 0;
	NVSDK_NGX_Parameter_GetI(
		sNgx.capabilities,
		NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver,
		&needsUpdatedDriver
	);
	int available = 0;
	NVSDK_NGX_Result const availableResult = (
		NVSDK_NGX_Parameter_GetI(
			sNgx.capabilities,
			NVSDK_NGX_Parameter_SuperSamplingDenoising_Available,
			&available
		)
	);
	if (needsUpdatedDriver != 0) {
		sNgx.rrUnavailableReason = "ray reconstruction needs a newer driver";
	} else if (NVSDK_NGX_FAILED(availableResult) || available == 0) {
		int initResultCode = 0;
		NVSDK_NGX_Parameter_GetI(
			sNgx.capabilities,
			NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult,
			&initResultCode
		);
		sNgx.rrUnavailableReason = (
			"ray reconstruction not available (feature init "
			+ result_string((NVSDK_NGX_Result)initResultCode) + ")"
		);
	} else {
		sNgx.rrAvailable = true;
		sNgx.rrUnavailableReason.clear();
	}
	printf(
		"[ngx] ray reconstruction %s%s\n",
		sNgx.rrAvailable ? "available" : "unavailable: ",
		sNgx.rrUnavailableReason.c_str()
	);
}

void vkof::ngx::shutdown() {
	if (!sNgx.initialized) { return; }
	ngx_rr_release();
	if (sNgx.capabilities != nullptr) {
		NVSDK_NGX_VULKAN_DestroyParameters(sNgx.capabilities);
	}
	NVSDK_NGX_VULKAN_Shutdown1(internal::native_device());
	sNgx = {};
}

bool vkof::ngx_rr_available() {
	return sNgx.rrAvailable;
}

char const * vkof::ngx_rr_unavailable_reason() {
	return sNgx.rrUnavailableReason.c_str();
}

u32v2 vkof::ngx_rr_render_size(
	NgxRrQuality const quality,
	u32v2 const outputSize
) {
	if (quality == NgxRrQuality::Dlaa) { return outputSize; }
	if (
		sNgx.sizeRender.x > 0u
		&& sNgx.sizeQuality == quality && sNgx.sizeOutput == outputSize
	) {
		return sNgx.sizeRender;
	}
	sNgx.sizeQuality = quality;
	sNgx.sizeOutput = outputSize;
	// NGX_DLSSD_GET_OPTIMAL_SETTINGS from the d3d helpers; vk has none
	void * callback = nullptr;
	if (sNgx.capabilities != nullptr) {
		NVSDK_NGX_Parameter_GetVoidPointer(
			sNgx.capabilities,
			NVSDK_NGX_Parameter_DLSSDOptimalSettingsCallback,
			&callback
		);
	}
	if (callback != nullptr) {
		NVSDK_NGX_Parameter * const params = sNgx.capabilities;
		NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_Width, outputSize.x);
		NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_Height, outputSize.y);
		NVSDK_NGX_Parameter_SetI(
			params, NVSDK_NGX_Parameter_PerfQualityValue, (int)quality
		);
		NVSDK_NGX_Parameter_SetI(params, NVSDK_NGX_Parameter_RTXValue, 0);
		auto const fn = (PFN_NVSDK_NGX_DLSS_GetOptimalSettingsCallback)callback;
		u32 width = 0u;
		u32 height = 0u;
		if (NVSDK_NGX_SUCCEED(fn(params))) {
			NVSDK_NGX_Parameter_GetUI(params, NVSDK_NGX_Parameter_OutWidth, &width);
			NVSDK_NGX_Parameter_GetUI(params, NVSDK_NGX_Parameter_OutHeight, &height);
		}
		if (width > 0u && height > 0u) {
			sNgx.sizeRender = u32v2 { width, height };
			return sNgx.sizeRender;
		}
	}
	// nvidia's published per-axis ratios
	f32 ratio = 1.0f;
	switch (quality) {
		case NgxRrQuality::Quality: ratio = 1.0f / 1.5f; break;
		case NgxRrQuality::Balanced: ratio = 0.58f; break;
		case NgxRrQuality::Performance: ratio = 0.5f; break;
		case NgxRrQuality::UltraPerformance: ratio = 1.0f / 3.0f; break;
		case NgxRrQuality::Dlaa: break;
	}
	sNgx.sizeRender = u32v2 {
		std::max((u32)((f32)outputSize.x * ratio + 0.5f), 1u),
		std::max((u32)((f32)outputSize.y * ratio + 0.5f), 1u),
	};
	return sNgx.sizeRender;
}

void vkof::ngx_rr_release() {
	if (sNgx.rr == nullptr) { return; }
	vkof::device_wait_idle();
	NVSDK_NGX_VULKAN_ReleaseFeature(sNgx.rr);
	NVSDK_NGX_VULKAN_DestroyParameters(sNgx.rrParameters);
	sNgx.rr = nullptr;
	sNgx.rrParameters = nullptr;
}

bool vkof::ngx_rr_evaluate(NgxRrEvaluateInfo const & info) {
	if (!sNgx.rrAvailable) { return false; }
	VkCommandBuffer const cmd = internal::native_command_buffer(info.cmd);

	// -- (re)create the feature
	bool const stale = (
		sNgx.rr != nullptr
		&& (
			sNgx.rrRenderSize != info.renderSize
			|| sNgx.rrOutputSize != info.outputSize
			|| sNgx.rrQuality != info.quality
			|| sNgx.rrPreset != info.preset
		)
	);
	if (stale) {
		ngx_rr_release();
	}
	if (sNgx.rr == nullptr) {
		NVSDK_NGX_VULKAN_AllocateParameters(&sNgx.rrParameters);
		// the hint is per quality mode; set them all
		for (char const * const hint : {
			NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA,
			NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality,
			NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced,
			NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance,
			NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance,
			NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraQuality,
		}) {
			NVSDK_NGX_Parameter_SetUI(sNgx.rrParameters, hint, (u32)info.preset);
		}
		auto createParams = NVSDK_NGX_DLSSD_Create_Params {};
		createParams.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
		createParams.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Unpacked;
		createParams.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_Linear;
		createParams.InWidth = info.renderSize.x;
		createParams.InHeight = info.renderSize.y;
		createParams.InTargetWidth = info.outputSize.x;
		createParams.InTargetHeight = info.outputSize.y;
		createParams.InPerfQualityValue = (
			(NVSDK_NGX_PerfQuality_Value)info.quality
		);
		createParams.InFeatureCreateFlags = (
			NVSDK_NGX_DLSS_Feature_Flags_IsHDR
			| NVSDK_NGX_DLSS_Feature_Flags_MVLowRes
		);
		NVSDK_NGX_Result const result = (
			NGX_VULKAN_CREATE_DLSSD_EXT1(
				internal::native_device(), cmd, 1u, 1u,
				&sNgx.rr, sNgx.rrParameters, &createParams
			)
		);
		internal::command_buffer_state_invalidate(info.cmd);
		if (NVSDK_NGX_FAILED(result)) {
			printf(
				"[ngx] ray reconstruction create failed: %s\n",
				result_string(result).c_str()
			);
			NVSDK_NGX_VULKAN_DestroyParameters(sNgx.rrParameters);
			sNgx.rr = nullptr;
			sNgx.rrParameters = nullptr;
			return false;
		}
		sNgx.rrRenderSize = info.renderSize;
		sNgx.rrOutputSize = info.outputSize;
		sNgx.rrQuality = info.quality;
		sNgx.rrPreset = info.preset;
		printf(
			"[ngx] ray reconstruction created at %ux%u -> %ux%u\n",
			info.renderSize.x, info.renderSize.y,
			info.outputSize.x, info.outputSize.y
		);
	}

	// -- evaluate
	NVSDK_NGX_Resource_VK color = resource(info.color, false);
	NVSDK_NGX_Resource_VK diffuseAlbedo = resource(info.diffuseAlbedo, false);
	NVSDK_NGX_Resource_VK specularAlbedo = resource(info.specularAlbedo, false);
	NVSDK_NGX_Resource_VK normals = resource(info.normals, false);
	NVSDK_NGX_Resource_VK roughness = resource(info.roughness, false);
	NVSDK_NGX_Resource_VK depth = resource(info.depth, false);
	NVSDK_NGX_Resource_VK motionVectors = resource(info.motionVectors, false);
	NVSDK_NGX_Resource_VK specularHitDistance = (
		resource(info.specularHitDistance, false)
	);
	NVSDK_NGX_Resource_VK output = resource(info.output, true);
	// the helper takes non-const pointers
	f32 worldToView[16];
	f32 viewToClip[16];
	std::copy(info.worldToView, info.worldToView + 16, worldToView);
	std::copy(info.viewToClip, info.viewToClip + 16, viewToClip);
	auto evalParams = NVSDK_NGX_VK_DLSSD_Eval_Params {};
	evalParams.pInDiffuseAlbedo = &diffuseAlbedo;
	evalParams.pInSpecularAlbedo = &specularAlbedo;
	evalParams.pInNormals = &normals;
	evalParams.pInRoughness = &roughness;
	evalParams.pInColor = &color;
	evalParams.pInOutput = &output;
	evalParams.pInDepth = &depth;
	evalParams.pInMotionVectors = &motionVectors;
	evalParams.pInSpecularHitDistance = &specularHitDistance;
	evalParams.pInWorldToViewMatrix = worldToView;
	evalParams.pInViewToClipMatrix = viewToClip;
	evalParams.InJitterOffsetX = info.jitter.x;
	evalParams.InJitterOffsetY = info.jitter.y;
	evalParams.InRenderSubrectDimensions = NVSDK_NGX_Dimensions {
		.Width = info.renderSize.x,
		.Height = info.renderSize.y,
	};
	evalParams.InReset = info.reset ? 1 : 0;
	evalParams.InMVScaleX = 1.0f;
	evalParams.InMVScaleY = 1.0f;
	NVSDK_NGX_Result const result = (
		NGX_VULKAN_EVALUATE_DLSSD_EXT(
			cmd, sNgx.rr, sNgx.rrParameters, &evalParams
		)
	);
	internal::command_buffer_state_invalidate(info.cmd);
	if (NVSDK_NGX_FAILED(result)) {
		printf(
			"[ngx] ray reconstruction evaluate failed: %s\n",
			result_string(result).c_str()
		);
		return false;
	}
	return true;
}
