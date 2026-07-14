#include <vkof/vkof.hpp>
#include <srat/core-handle.hpp>
#include <srat/profile.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>

#include <stb_image_write.h>

#if defined(VKOF_AFTERMATH)
#include "aftermath.hpp"
#endif

int vkof_write_png_uncompressed(
	char const * path, int w, int h, int comp,
	void const * data, int stride
);

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>

#define VkAssert(x) { \
	VkResult const vkr_ = (x); \
	if (vkr_ != VK_SUCCESS) { \
		if (vkr_ == VK_ERROR_DEVICE_LOST) { \
			vkof_aftermath_on_device_lost(); \
		} \
		printf("assertion failed: %s [%d]\n", #x, (int)vkr_); \
		std::abort(); \
	} \
}

#if !defined(VKOF_AFTERMATH)
static void vkof_aftermath_on_device_lost() {}
#endif

// -----------------------------------------------------------------------------
// -- shader probe
// -----------------------------------------------------------------------------

static std::string sProbeMessage;
static bool sShaderReloaded = false;

static VKAPI_ATTR VkBool32 VKAPI_CALL debugMessengerCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT const severity,
	VkDebugUtilsMessageTypeFlagsEXT,
	VkDebugUtilsMessengerCallbackDataEXT const * const data,
	void *
) {
	if (data->pMessageIdName && strstr(data->pMessageIdName, "DEBUG-PRINTF")) {
		// strip "...DebugPrintf:\n" prefix, keep only the user string
		char const * found = nullptr;
		if (data->pMessage) {
			char const * marker = strstr(data->pMessage, "DebugPrintf:\n");
			if (marker) {
				found = marker + 13u;
			}
		}
		if (found) {
			sProbeMessage = found;
		}
		return VK_FALSE;
	}
	if (
		data->pMessage
		&& strstr(data->pMessage, "DebugPrintf logs to the Information")
	) {
		return VK_FALSE;
	}
	if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
		return VK_FALSE;
	}
	fprintf(stderr, "[vkof] %s\n", data->pMessage);
	return VK_FALSE;
}

char const * vkof::probe_message() {
	if (sProbeMessage.empty()) { return nullptr; }
	return sProbeMessage.c_str();
}

bool vkof::shader_reloaded() { return sShaderReloaded; }

// -----------------------------------------------------------------------------
// -- private forwards
// -----------------------------------------------------------------------------

namespace {
	struct ImplTexture;
	struct ImplPipelineGraphics;
	struct ImplPipelineCompute;
	struct CompileResult;

	enum struct ShaderStage { fragment, compute, mesh };
	struct CompileResult {
		int exitCode;
		std::string output;
	};

	struct ImplCommandBuffer
	{
		VkCommandBuffer cmd;
		vkof::Pipeline boundPipeline;
		VkPipelineBindPoint bindPoint;
	};

	struct ImplAccelerationStructureBlas
	{
		VkAccelerationStructureKHR handle { VK_NULL_HANDLE };
		VkBuffer backingBuffer { VK_NULL_HANDLE };
		VmaAllocation backingAlloc { nullptr };
		u64 deviceAddress { 0u };
	};

	struct ImplAccelerationStructureTlas
	{
		VkAccelerationStructureKHR handle { VK_NULL_HANDLE };
		VkBuffer backingBuffer { VK_NULL_HANDLE };
		VmaAllocation backingAlloc { nullptr };
		u64 deviceAddress { 0u };
		VkBuffer scratchBuffer { VK_NULL_HANDLE };
		VmaAllocation scratchAlloc { nullptr };
		u64 scratchVa { 0u };
		VkBuffer instanceBuffer { VK_NULL_HANDLE };
		VmaAllocation instanceAlloc { nullptr };
		void * instanceMapped { nullptr };
		u64 instanceVa { 0u };
		u32 maxInstances { 0u };
	};
}

static std::filesystem::file_time_type file_write_time(std::string const & p);

static VkImageView image_storage_view_for_mip(
	ImplTexture const & implTexture, u32 mipLevel
);
static char const * shader_stage_flag(ShaderStage const & stage);
static CompileResult system_capture(char const * cmd);
static VkShaderModule compile_and_load_shader_module(
	std::string const & glslPath,
	ShaderStage const stage,
	std::vector<std::string> const & defines,
	std::vector<std::string> const & includePaths
);
static VkPipeline build_debug_line_pipeline(
	std::string const & meshPath,
	std::string const & fragPath,
	bool const withDepth
);
static bool file_changed(
	std::string const & path,
	std::filesystem::file_time_type const & lastWriteTime
);
static void pipeline_hot_reload();
static void profiler_init();

static VkFormat to_vk_format(vkof::ImageFormat format) {
	switch (format) {
		case vkof::ImageFormat::r32ui:
			return VK_FORMAT_R32_UINT;
		case vkof::ImageFormat::r8g8b8a8_unorm:
			return VK_FORMAT_R8G8B8A8_UNORM;
		case vkof::ImageFormat::r8g8b8a8_srgb:
			return VK_FORMAT_R8G8B8A8_SRGB;
		case vkof::ImageFormat::b8g8r8a8_unorm:
			return VK_FORMAT_B8G8R8A8_UNORM;
		case vkof::ImageFormat::r16g16b16a16_sfloat:
			return VK_FORMAT_R16G16B16A16_SFLOAT;
		case vkof::ImageFormat::r32_float:
			return VK_FORMAT_R32_SFLOAT;
		case vkof::ImageFormat::r32g32b32a32_sfloat:
			return VK_FORMAT_R32G32B32A32_SFLOAT;
		case vkof::ImageFormat::d24_unorm_s8_uint:
			return VK_FORMAT_D24_UNORM_S8_UINT;
		default:
			return VK_FORMAT_UNDEFINED;
	}
}

static VkCullModeFlagBits to_vk_cull(vkof::CullMode cullMode) {
	switch (cullMode) {
		case vkof::CullMode::front:
			return VK_CULL_MODE_FRONT_BIT;
		case vkof::CullMode::back:
			return VK_CULL_MODE_BACK_BIT;
		default:
			return VK_CULL_MODE_NONE;
	}
}

static void depth_test_flags(
	vkof::DepthTest const writeOnTestOff,
	VkBool32 & outDepthTestEnable,
	VkBool32 & outDepthWriteEnable
) {
	switch (writeOnTestOff) {
		case vkof::DepthTest::write_off_test_off:
			outDepthTestEnable = VK_FALSE;
			outDepthWriteEnable = VK_FALSE;
		break;
		case vkof::DepthTest::write_on_test_off:
			outDepthTestEnable = VK_FALSE;
			outDepthWriteEnable = VK_TRUE;
		break;
		case vkof::DepthTest::write_off_test_on:
			outDepthTestEnable = VK_TRUE;
			outDepthWriteEnable = VK_FALSE;
		break;
		case vkof::DepthTest::write_on_test_on:
			outDepthTestEnable = VK_TRUE;
			outDepthWriteEnable = VK_TRUE;
		break;
	}
}

namespace
{
	// derive the aspect mask from a format (color vs depth)
	VkImageAspectFlags format_aspect(VkFormat const format) {
		switch (format) {
			case VK_FORMAT_D32_SFLOAT:
			case VK_FORMAT_D16_UNORM:
				return VK_IMAGE_ASPECT_DEPTH_BIT;
			case VK_FORMAT_D24_UNORM_S8_UINT:
			case VK_FORMAT_D32_SFLOAT_S8_UINT:
				return (
					VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
				);
			default:
				return VK_IMAGE_ASPECT_COLOR_BIT;
		}
	}

	// usage depends on whether the format is depth or color
	VkImageUsageFlags format_usage(
		VkPhysicalDevice const physicalDevice,
		VkFormat const format
	) {
		VkImageUsageFlags usage = (
			  VK_IMAGE_USAGE_SAMPLED_BIT
			| VK_IMAGE_USAGE_TRANSFER_SRC_BIT
			| VK_IMAGE_USAGE_TRANSFER_DST_BIT
		);
		bool const isDepth = (
			format_aspect(format) & VK_IMAGE_ASPECT_DEPTH_BIT
		);
		if (isDepth) {
			usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		} else {
			usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
			VkFormatProperties props {};
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
			if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) {
				usage |= VK_IMAGE_USAGE_STORAGE_BIT;
			}
		}
		return usage;
	}

	struct ImplTexture
	{
		VkImage image;
		VmaAllocation allocation;
		VkImageView imageViewFull;
		std::vector<VkImageView> imageViewPerMip;
		u32 width;
		u32 height;
		u32 depth;
		VkFormat vkFormat;
	};

	struct ImplBuffer
	{
		VkBuffer buffer;
		VmaAllocation allocation;
		u64 deviceAddress;  // Va, present for every buffer
		void * mapped;      // host pointer, HostWritable only (else null)
		u64 byteCount;      // size of the allocation in bytes
	};

	struct ImplSampler
	{
		VkSampler sampler;
	};

	struct ImplTransientImage {
		vkof::ImageFormat format;
		f32 scaleWidth;
		f32 scaleHeight;
		u32 mipLevels;
		bool isDoubleBuffered;
		// pre-allocated image handles (index 0 always valid; index 1 only if
		// isDoubleBuffered).  stored in imagePool so all image_* helpers work.
		std::array<vkof::Image, 2> handles;
	};

	struct ImplTransientBuffer {
		bool isDoubleBuffered;
		// pre-allocated buffer handles (index 1 only if isDoubleBuffered)
		std::array<vkof::Buffer, 2> handles;
	};

	struct ImplRenderBarrier
	{
		vkof::TransientImage transientImage;
		vkof::RenderNodeAccess access;
	};

	struct ImplBufferBarrier
	{
		vkof::TransientBuffer transientBuffer;
		vkof::RenderNodeAccess access;
	};

	struct ImplColorAttachment
	{
		vkof::TransientImage image;
		vkof::RenderNodeLoadOp loadOp;
		u32 mipLevel;
		u32 colorIndex;
		f32 clearColor[4];
	};

	struct ImplDepthAttachment
	{
		vkof::TransientImage image;
		vkof::RenderNodeLoadOp loadOp;
		u32 mipLevel;
		f32 clearDepth;
	};

	struct ImplRenderNode
	{
		vkof::CommandQueue queue;
		std::vector<ImplRenderBarrier> imageBarriers;
		std::vector<ImplBufferBarrier> bufferBarriers;
		std::vector<vkof::Image> persistentImages;
		std::vector<ImplColorAttachment> colorAttachments;
		std::optional<ImplDepthAttachment> depthAttachment;
		std::function<void(vkof::CommandBuffer const &)> callback;
	};

	struct ImplPipelineGraphics
	{
		VkPipeline pipeline;
		std::vector<vkof::ImageFormat> colorAttachmentFormats;
		vkof::ImageFormat depthAttachmentFormat;
		vkof::DepthTest depthTest;
		vkof::CullMode cullMode;
		vkof::BlendMode blendMode;

		std::string pathFragment;
		std::string pathMesh;
		std::vector<std::string> defines;
		std::vector<std::string> includePaths;

		std::filesystem::file_time_type lastWriteTimeFragment;
		std::filesystem::file_time_type lastWriteTimeMesh;

		std::vector<std::string> dependencyPaths;
		std::vector<std::filesystem::file_time_type> dependencyWriteTimes;
	};

	struct ImplPipelineCompute
	{
		VkPipeline pipeline;
		VkPipelineLayout layout;
		std::string pathCompute;
		std::vector<std::string> defines;
		std::vector<std::string> includePaths;
		std::filesystem::file_time_type lastWriteTimeCompute;

		std::vector<std::string> dependencyPaths;
		std::vector<std::filesystem::file_time_type> dependencyWriteTimes;
	};

	struct PerFrameData {
		VkCommandBuffer cmdGraphics;
		VkSemaphore semaphoreAcquire;
		VkFence fence;
	};

	static constexpr u32 kFramesInFlight = 2;
	static constexpr u32 kMaxProfiledNodes = 16;

	struct NodeTiming {
		double cpuMs { 0.0 };
		double gpuMs { 0.0 };
	};

	struct Profiler {
		VkQueryPool queryPool { VK_NULL_HANDLE };
		double timestampPeriodMs { 0.0 };
		bool gpuSupported { false };
		// how many nodes ran in the last use of each frame slot (for readback)
		std::array<u32, kFramesInFlight> slotNodeCounts {};
		// stable display data: gpu updated on readback, cpu updated each frame
		std::array<NodeTiming, kMaxProfiledNodes> timings {};
		u32 displayCount { 0 };
		u32 frameNodeCount { 0 };
	};

	struct ImplDevice {
		GLFWwindow * window;
		vkb::Instance instance;
		vkb::PhysicalDevice physicalDevice;
		vkb::Device device;
		vkb::Swapchain swapchain;
		std::vector<VkImage> swapchainImages;
		std::vector<VkImageView> swapchainImageViews;
		std::vector<VkImage> swapchainDepthImages;
		std::vector<VkImageView> swapchainDepthImageViews;
		std::vector<VmaAllocation> swapchainDepthAllocs;
		// non-null only in headless mode;
		//   holds the fake swapchain image allocations
		VmaAllocation headlessSwapchainAlloc { VK_NULL_HANDLE };
		VmaAllocation headlessDepthAlloc { VK_NULL_HANDLE };

		u32 swapchainImageIndex;

		// internal render resolution; transient images are sized from this,
		//   decoupled from the swapchain extent
		VkExtent2D renderExtent;

		VkCommandPool commandPoolGraphics;
		VkCommandPool commandPoolCompute;

		VkPipelineLayout pipelineLayoutUniversal;

		VkSurfaceKHR surface;
		VkQueue queueGraphics;
		VkQueue queueCompute;
		VmaAllocator allocator;

		// per-frame-in-flight sync primitives
		std::array<PerFrameData, kFramesInFlight> frameData;
		u32 frameIndex { 0 };
		// per-swapchain-image render semaphores. must not be per-frame-slot;
		//   reusing before WSI releases them triggers
		//   VUID-vkQueueSubmit2-semaphore-03868)
		std::vector<VkSemaphore> semaphoresRender;

		srat::HandlePool<vkof::Image, ImplTexture> imagePool;
		srat::HandlePool<vkof::Buffer, ImplBuffer> bufferPool;
		srat::HandlePool<vkof::Sampler, ImplSampler> samplerPool;
		srat::HandlePool<vkof::TransientImage, ImplTransientImage>
			transientImagePool;
		srat::HandlePool<vkof::TransientBuffer, ImplTransientBuffer>
			transientBufferPool;
		srat::HandlePool<vkof::Pipeline, ImplPipelineGraphics>
			pipelineGraphicsPool;
		srat::HandlePool<vkof::Pipeline, ImplPipelineCompute>
			pipelineComputePool;
		srat::HandlePool<vkof::CommandBuffer, ImplCommandBuffer>
			commandBufferPool;
		srat::HandlePool<vkof::RenderNode, ImplRenderNode>
			renderNodePool;
		srat::HandlePool<
			vkof::AccelerationStructureBlas, ImplAccelerationStructureBlas
		> blasPool;
		srat::HandlePool<
			vkof::AccelerationStructureTlas, ImplAccelerationStructureTlas
		> tlasPool;

		VkDescriptorPool imguiDescriptorPool { VK_NULL_HANDLE };
		bool imguiFrameActive { false };
		Profiler profiler;

		// need a linear list of pipeline handles for hot reload iteration
		std::vector<vkof::Pipeline> pipelineGraphicsHandles {};
		std::vector<vkof::Pipeline> pipelineComputeHandles {};

		// linear list of live transient images for render_extent_set iteration
		//   and cleanup in shutdown
		std::vector<vkof::TransientImage> transientImageHandles {};
		// transient buffer resource handles for cleanup in shutdown
		std::vector<vkof::Buffer> transientBufferCleanupHandles {};

#if defined(VKOF_AFTERMATH)
		PFN_vkCmdSetCheckpointNV pfnCmdSetCheckpointNV { nullptr };
		PFN_vkGetQueueCheckpointDataNV pfnGetQueueCheckpointDataNV { nullptr };
#endif
		PFN_vkCreateAccelerationStructureKHR pfnCreateAccelerationStructureKHR
			{ nullptr };
		PFN_vkDestroyAccelerationStructureKHR pfnDestroyAccelerationStructureKHR
			{ nullptr };
		PFN_vkGetAccelerationStructureBuildSizesKHR
			pfnGetAccelerationStructureBuildSizesKHR { nullptr };
		PFN_vkGetAccelerationStructureDeviceAddressKHR
			pfnGetAccelerationStructureDeviceAddressKHR { nullptr };
		PFN_vkCmdBuildAccelerationStructuresKHR
			pfnCmdBuildAccelerationStructuresKHR { nullptr };
	};
}
static std::filesystem::file_time_type file_write_time(std::string const & p) {
	std::error_code ec;
	auto const t = std::filesystem::last_write_time(p, ec);
	return ec ? std::filesystem::file_time_type {} : t;
}

static ImplDevice * sDevice;

static PFN_vkCmdDrawMeshTasksEXT pfnVkCmdDrawMeshTasksEXT;
static PFN_vkSetDebugUtilsObjectNameEXT pfnSetDebugName;

static void set_debug_name(
	VkObjectType type,
	u64 handle,
	std::source_location const & loc
) {
	if (!pfnSetDebugName) { return; }
	std::string const path = loc.file_name();
	std::string const file = path.substr(path.rfind('/') + 1u);
	std::string const name = file + ":" + std::to_string(loc.line());
	VkDebugUtilsObjectNameInfoEXT const info = {
		.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
		.pNext = nullptr,
		.objectType = type,
		.objectHandle = handle,
		.pObjectName = name.c_str(),
	};
	pfnSetDebugName(sDevice->device, &info);
}

static VkDescriptorSetLayout sBindlessSetLayout = VK_NULL_HANDLE;
static VkDescriptorPool sBindlessPool = VK_NULL_HANDLE;
static VkDescriptorSet sBindlessSet = VK_NULL_HANDLE;
static u32 sBindlessSamplerNextSlot = 1u;
static u32 sBindlessStorageUintNextSlot = 1u;
static u32 sBindlessStorageFloatNextSlot = 1u;
static u32 sBindlessSampler3dNextSlot = 1u;
static u32 sBindlessStorage3dNextSlot = 1u;
static constexpr u32 skBindlessMaxSlots = 65536u;
// slots released by image_destroy, reused before growing their counter;
//   kept per-binding since each binding is a separate descriptor array
static std::vector<u32> sBindlessSamplerFreeSlots;
static std::vector<u32> sBindlessStorageUintFreeSlots;
static std::vector<u32> sBindlessStorageFloatFreeSlots;
static std::vector<u32> sBindlessSampler3dFreeSlots;
static std::vector<u32> sBindlessStorage3dFreeSlots;
// image id -> (binding, slot) for every bindless handle ever allocated
//   against it; image_destroy consults this to release its slots back to
//   the free lists above instead of leaving stale descriptor entries
//   pointing at a destroyed VkImageView forever
using BindlessSlotList = std::vector<std::pair<u32, u32>>;
static std::unordered_map<u64, BindlessSlotList> sImageBindlessSlots;

static u32 alloc_bindless_slot(std::vector<u32> & freeList, u32 & counter)
{
	if (!freeList.empty()) {
		u32 const slot = freeList.back();
		freeList.pop_back();
		return slot;
	}
	return counter++;
}

static void write_storage_image_descriptor(
	ImplTexture const & implTexture,
	u32 const mipLevel,
	u32 const binding,
	u32 const slot
) {
	VkImageView const view = (
		mipLevel == 0u
		? implTexture.imageViewFull
		: image_storage_view_for_mip(implTexture, mipLevel)
	);
	VkDescriptorImageInfo const imageInfo = {
		.sampler = VK_NULL_HANDLE,
		.imageView = view,
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
	};
	VkWriteDescriptorSet const write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.pNext = nullptr,
		.dstSet = sBindlessSet,
		.dstBinding = binding,
		.dstArrayElement = slot,
		.descriptorCount = 1u,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.pImageInfo = &imageInfo,
		.pBufferInfo = nullptr,
		.pTexelBufferView = nullptr,
	};
	vkUpdateDescriptorSets(sDevice->device, 1u, &write, 0u, nullptr);
}

// key covers the ping-pong slot too: each physical image backing a
//   double-buffered TransientImage gets its own bindless slot, written once
//   and left alone, so a still-in-flight frame reading through the other
//   ping-pong slot is never repointed out from under it. render_extent_set
//   rewrites the view in place (same slot number) when it recreates images
static u64 transient_storage_cache_key(
	u64 const transientImageId, u32 const mipLevel, u32 const pingPongSlot
) {
	return (
		(transientImageId << 16u) | (u64(mipLevel) << 1u) | u64(pingPongSlot)
	);
}

static std::unordered_map<u64, u32> sTransientStorageSlotCache;

struct DebugVertex {
	f32v3 pos;
	f32v3 color;
};

struct DebugDrawPC {
	u64 vertexVa;
	f32m44 viewProj;
};

static constexpr u32 skDebugMaxLines = 65536u;
static std::vector<DebugVertex> sDebugVertices;
static VkPipeline sDebugLinePipelineWithDepth = VK_NULL_HANDLE;
static VkPipeline sDebugLinePipelineNoDepth = VK_NULL_HANDLE;
static vkof::Buffer sDebugVertexBuffer { 0 };

struct DebugSpherePC {
	u64 sphereVa;
	f32m44 viewProj;
	u32 startIdx;
};

struct DebugSphereBatch {
	u32 startIdx;
	u32 count;
	vkof::Pipeline pipeline;
};

static constexpr u32 skDebugSphereInitialCapacity = 4096u;
static u32 sDebugSphereCapacity = skDebugSphereInitialCapacity;
static std::vector<vkof::DebugSphere> sDebugSphereData;
static std::vector<DebugSphereBatch> sDebugSphereBatches;
static vkof::Buffer sDebugSphereBuffer { 0 };

// -----------------------------------------------------------------------------
// -- debug draw
// -----------------------------------------------------------------------------

void vkof::debug_draw_line(
	f32v3 const & a, f32v3 const & b, f32v3 const & color
) {
	if (sDebugVertices.size() >= skDebugMaxLines * 2u) { return; }
	sDebugVertices.push_back({ .pos = a, .color = color });
	sDebugVertices.push_back({ .pos = b, .color = color });
}

void vkof::debug_draw_box(
	f32v3 const & min, f32v3 const & max, f32v3 const & color
) {
	debug_draw_line({ min.x, min.y, min.z }, { max.x, min.y, min.z }, color);
	debug_draw_line({ max.x, min.y, min.z }, { max.x, min.y, max.z }, color);
	debug_draw_line({ max.x, min.y, max.z }, { min.x, min.y, max.z }, color);
	debug_draw_line({ min.x, min.y, max.z }, { min.x, min.y, min.z }, color);
	debug_draw_line({ min.x, max.y, min.z }, { max.x, max.y, min.z }, color);
	debug_draw_line({ max.x, max.y, min.z }, { max.x, max.y, max.z }, color);
	debug_draw_line({ max.x, max.y, max.z }, { min.x, max.y, max.z }, color);
	debug_draw_line({ min.x, max.y, max.z }, { min.x, max.y, min.z }, color);
	debug_draw_line({ min.x, min.y, min.z }, { min.x, max.y, min.z }, color);
	debug_draw_line({ max.x, min.y, min.z }, { max.x, max.y, min.z }, color);
	debug_draw_line({ max.x, min.y, max.z }, { max.x, max.y, max.z }, color);
	debug_draw_line({ min.x, min.y, max.z }, { min.x, max.y, max.z }, color);
}

void vkof::debug_draw_sphere(
	f32v3 const & center, f32 const radius, f32v3 const & color
) {
	static constexpr u32 kSegments = 16u;
	static constexpr f32 kTwoPi = 6.28318530f;
	for (u32 i = 0u; i < kSegments; ++i) {
		f32 const t0 = ((f32)i / (f32)kSegments) * kTwoPi;
		f32 const t1 = ((f32)(i + 1u) / (f32)kSegments) * kTwoPi;
		f32 const c0 = cosf(t0);
		f32 const s0 = sinf(t0);
		f32 const c1 = cosf(t1);
		f32 const s1 = sinf(t1);
		debug_draw_line(
			{ center.x + radius * c0, center.y + radius * s0, center.z },
			{ center.x + radius * c1, center.y + radius * s1, center.z },
			color
		);
		debug_draw_line(
			{ center.x + radius * c0, center.y, center.z + radius * s0 },
			{ center.x + radius * c1, center.y, center.z + radius * s1 },
			color
		);
		debug_draw_line(
			{ center.x, center.y + radius * c0, center.z + radius * s0 },
			{ center.x, center.y + radius * c1, center.z + radius * s1 },
			color
		);
	}
}

static bool format_is_uint(VkFormat const fmt) {
	switch (fmt) {
		case VK_FORMAT_R32_UINT:
		case VK_FORMAT_R32G32_UINT:
		case VK_FORMAT_R32G32B32A32_UINT:
		case VK_FORMAT_R16_UINT:
		case VK_FORMAT_R16G16_UINT:
		case VK_FORMAT_R16G16B16A16_UINT:
		case VK_FORMAT_R8_UINT:
		case VK_FORMAT_R8G8_UINT:
		case VK_FORMAT_R8G8B8A8_UINT:
			return true;
		default:
			return false;
	}
}

static VkDescriptorSetLayout create_bindless_set_layout(VkDevice const device)
{
	VkDescriptorBindingFlags const skBoundAndUpdateable = (
		VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
		| VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
	);
	VkDescriptorSetLayoutBinding const bindings[] = {
		{
			// binding 0: sampler2D
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = skBindlessMaxSlots,
			.stageFlags = VK_SHADER_STAGE_ALL,
			.pImmutableSamplers = nullptr,
		},
		{
			// binding 1: uimage2D (uint storage)
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = skBindlessMaxSlots,
			.stageFlags = VK_SHADER_STAGE_ALL,
			.pImmutableSamplers = nullptr,
		},
		{
			// binding 2: image2D (float storage)
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = skBindlessMaxSlots,
			.stageFlags = VK_SHADER_STAGE_ALL,
			.pImmutableSamplers = nullptr,
		},
		{
			// binding 3: single acceleration structure slot (TLAS)
			.binding = 3,
			.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
			.descriptorCount = 1u,
			.stageFlags = VK_SHADER_STAGE_ALL,
			.pImmutableSamplers = nullptr,
		},
		{
			// binding 4: sampler3D
			.binding = 4,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = skBindlessMaxSlots,
			.stageFlags = VK_SHADER_STAGE_ALL,
			.pImmutableSamplers = nullptr,
		},
		{
			// binding 5: image3D (float storage)
			.binding = 5,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = skBindlessMaxSlots,
			.stageFlags = VK_SHADER_STAGE_ALL,
			.pImmutableSamplers = nullptr,
		},
	};
	VkDescriptorBindingFlags const bindingFlags[] = {
		skBoundAndUpdateable,
		skBoundAndUpdateable,
		skBoundAndUpdateable,
		skBoundAndUpdateable,
		skBoundAndUpdateable,
		skBoundAndUpdateable,
	};
	VkDescriptorSetLayoutBindingFlagsCreateInfo const bindingFlagsInfo = {
		.sType = (
			VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO
		),
		.pNext = nullptr,
		.bindingCount = 6u,
		.pBindingFlags = bindingFlags,
	};
	VkDescriptorSetLayoutCreateInfo const ci = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext = &bindingFlagsInfo,
		.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		.bindingCount = 6u,
		.pBindings = bindings,
	};
	VkDescriptorSetLayout layout;
	VkAssert(vkCreateDescriptorSetLayout(device, &ci, nullptr, &layout));
	return layout;
}

static void create_bindless_pool_and_set(VkDevice const device)
{
	VkDescriptorPoolSize const poolSizes[] = {
		{
			// bindings 0 (sampler2D) + 4 (sampler3D)
			.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = skBindlessMaxSlots * 2u,
		},
		{
			// bindings 1 (uint image2D) + 2 (float image2D) + 5 (float image3D)
			.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = skBindlessMaxSlots * 3u,
		},
		{
			// binding 3: single TLAS slot
			.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
			.descriptorCount = 1u,
		},
	};
	VkDescriptorPoolCreateInfo const poolCi = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.pNext = nullptr,
		.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
		.maxSets = 1u,
		.poolSizeCount = 3u,
		.pPoolSizes = poolSizes,
	};
	VkAssert(vkCreateDescriptorPool(device, &poolCi, nullptr, &sBindlessPool));

	VkDescriptorSetAllocateInfo const allocInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.pNext = nullptr,
		.descriptorPool = sBindlessPool,
		.descriptorSetCount = 1u,
		.pSetLayouts = &sBindlessSetLayout,
	};
	VkAssert(vkAllocateDescriptorSets(device, &allocInfo, &sBindlessSet));
}

// -----------------------------------------------------------------------------
// -- buffer
// -----------------------------------------------------------------------------

vkof::Buffer vkof::buffer_create(
	vkof::BufferCreateInfo const & ci,
	std::source_location loc
) {
	ImplBuffer implBuffer;
	VkBufferCreateInfo const bufferCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.size = ci.byteCount,
		.usage = (
			  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
			| VK_BUFFER_USAGE_INDEX_BUFFER_BIT
			| VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
			| VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
			| VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
			| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
			| VK_BUFFER_USAGE_TRANSFER_SRC_BIT
			| VK_BUFFER_USAGE_TRANSFER_DST_BIT
			| VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
		),
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = nullptr,
	};
	bool const isHost = (ci.memory == vkof::BufferMemory::HostWritable);
	VmaAllocationCreateInfo const allocationCreateInfo = {
		.flags = (
			VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
			| (
				isHost
					? (VmaAllocationCreateFlags)(
						  VMA_ALLOCATION_CREATE_MAPPED_BIT
						| VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
					)
					: (VmaAllocationCreateFlags)0
			)
		),
		.usage = (
			isHost
				? VMA_MEMORY_USAGE_AUTO_PREFER_HOST
				: VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
		),
		.requiredFlags = 0,
		.preferredFlags = 0,
		.memoryTypeBits = 0,
		.pool = nullptr,
		.pUserData = nullptr,
		.priority = 0.0f,
	};
	VmaAllocationInfo allocationInfo;
	VkAssert(
		vmaCreateBuffer(
			sDevice->allocator,
			&bufferCreateInfo,
			&allocationCreateInfo,
			&implBuffer.buffer,
			&implBuffer.allocation,
			&allocationInfo
		)
	);

	// -- Va, fetched for every buffer (the core of the binding model)
	VkBufferDeviceAddressInfo const addressInfo = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		.pNext = nullptr,
		.buffer = implBuffer.buffer,
	};
	implBuffer.deviceAddress = (
		vkGetBufferDeviceAddress(sDevice->device, &addressInfo)
	);

	// -- host mapped pointer, only for host-writable (VMA mapped it for us)
	implBuffer.mapped = (isHost ? allocationInfo.pMappedData : nullptr);
	implBuffer.byteCount = ci.byteCount;

	vkof::Buffer const handle = sDevice->bufferPool.allocate(implBuffer);
	set_debug_name(
		VK_OBJECT_TYPE_BUFFER,
		(u64)implBuffer.buffer,
		loc
	);
	return handle;
}

void vkof::buffer_destroy(Buffer const & buffer)
{
	auto const implBuffer = sDevice->bufferPool.get(buffer);
	if (implBuffer) {
		vmaDestroyBuffer(
			sDevice->allocator, implBuffer->buffer, implBuffer->allocation
		);
		sDevice->bufferPool.free(buffer);
	}
}

u64 vkof::buffer_virtual_address(Buffer const & buffer)
{
	auto const implBuffer = sDevice->bufferPool.get(buffer);
	if (implBuffer) {
		return implBuffer->deviceAddress;
	}
	return 0;
}

srat::slice<u8> vkof::buffer_host_address(Buffer const & buffer)
{
	auto const implBuffer = sDevice->bufferPool.get(buffer);
	if (implBuffer && implBuffer->mapped != nullptr) {
		return srat::slice<u8>(
			(u8 *)implBuffer->mapped, implBuffer->byteCount
		);
	}
	return srat::slice<u8>(nullptr, 0);
}

void vkof::buffer_upload(BufferUploadInfo const & uploadInfo)
{
	auto const dst = sDevice->bufferPool.get(uploadInfo.buffer);
	if (!dst || uploadInfo.data.size() == 0) {
		return;
	}

	// -- staging buffer (host-visible, mapped)
	VkBufferCreateInfo const stagingCi = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.size = uploadInfo.data.size(),
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = nullptr,
	};
	VmaAllocationCreateInfo const stagingAllocCi = {
		.flags = (
			  VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
			| VMA_ALLOCATION_CREATE_MAPPED_BIT
			| VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		),
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
		.requiredFlags = 0,
		.preferredFlags = 0,
		.memoryTypeBits = 0,
		.pool = nullptr,
		.pUserData = nullptr,
		.priority = 0.0f,
	};
	VkBuffer stagingBuffer;
	VmaAllocation stagingAlloc;
	VmaAllocationInfo stagingInfo;
	VkAssert(
		vmaCreateBuffer(
			sDevice->allocator,
			&stagingCi,
			&stagingAllocCi,
			&stagingBuffer,
			&stagingAlloc,
			&stagingInfo
		)
	);
	std::memcpy(
		stagingInfo.pMappedData,
		uploadInfo.data.ptr(),
		uploadInfo.data.size()
	);

	// -- one-shot command buffer
	VkCommandBufferAllocateInfo const cbai = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = nullptr,
		.commandPool = sDevice->commandPoolGraphics,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer cmd;
	VkAssert(vkAllocateCommandBuffers(sDevice->device, &cbai, &cmd));
	VkCommandBufferBeginInfo const beginInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = nullptr,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = nullptr,
	};
	VkAssert(vkBeginCommandBuffer(cmd, &beginInfo));

	// in-flight frames may still be reading the destination (write-after-read)
	VkBufferMemoryBarrier const barrierBefore = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.pNext = nullptr,
		.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer = dst->buffer,
		.offset = uploadInfo.byteOffset,
		.size = uploadInfo.data.size(),
	};
	vkCmdPipelineBarrier(
		cmd,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, 0, nullptr, 1, &barrierBefore, 0, nullptr
	);

	VkBufferCopy const region = {
		.srcOffset = 0,
		.dstOffset = uploadInfo.byteOffset,
		.size = uploadInfo.data.size(),
	};
	vkCmdCopyBuffer(cmd, stagingBuffer, dst->buffer, 1, &region);

	// make the copy visible to subsequent reads
	VkBufferMemoryBarrier const barrierAfter = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.pNext = nullptr,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer = dst->buffer,
		.offset = uploadInfo.byteOffset,
		.size = uploadInfo.data.size(),
	};
	vkCmdPipelineBarrier(
		cmd,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		0, 0, nullptr, 1, &barrierAfter, 0, nullptr
	);

	VkAssert(vkEndCommandBuffer(cmd));

	VkSubmitInfo const submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = nullptr,
		.waitSemaphoreCount = 0,
		.pWaitSemaphores = nullptr,
		.pWaitDstStageMask = nullptr,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd,
		.signalSemaphoreCount = 0,
		.pSignalSemaphores = nullptr,
	};
	VkAssert(
		vkQueueSubmit(sDevice->queueGraphics, 1, &submitInfo, VK_NULL_HANDLE)
	);
	VkAssert(vkQueueWaitIdle(sDevice->queueGraphics));

	vkFreeCommandBuffers(
		sDevice->device, sDevice->commandPoolGraphics, 1, &cmd
	);
	vmaDestroyBuffer(sDevice->allocator, stagingBuffer, stagingAlloc);
}

void vkof::buffer_download(BufferDownloadInfo const & info)
{
	auto const src = sDevice->bufferPool.get(info.buffer);
	if (!src || info.dst.size() == 0) {
		return;
	}

	// -- host-visible staging buffer (destination of the GPU copy)
	VkBufferCreateInfo const stagingCi = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.size = info.dst.size(),
		.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = nullptr,
	};
	VmaAllocationCreateInfo const stagingAllocCi = {
		.flags = (
			  VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
			| VMA_ALLOCATION_CREATE_MAPPED_BIT
			| VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
		),
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
		.requiredFlags = 0,
		.preferredFlags = 0,
		.memoryTypeBits = 0,
		.pool = nullptr,
		.pUserData = nullptr,
		.priority = 0.0f,
	};
	VkBuffer stagingBuffer;
	VmaAllocation stagingAlloc;
	VmaAllocationInfo stagingInfo;
	VkAssert(
		vmaCreateBuffer(
			sDevice->allocator,
			&stagingCi,
			&stagingAllocCi,
			&stagingBuffer,
			&stagingAlloc,
			&stagingInfo
		)
	);

	// -- one-shot command buffer: GPU buffer -> staging
	VkCommandBufferAllocateInfo const cbai = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = nullptr,
		.commandPool = sDevice->commandPoolGraphics,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer cmd;
	VkAssert(vkAllocateCommandBuffers(sDevice->device, &cbai, &cmd));
	VkCommandBufferBeginInfo const beginInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = nullptr,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = nullptr,
	};
	VkAssert(vkBeginCommandBuffer(cmd, &beginInfo));

	// memory barrier: make sure all prior writes are visible before the copy
	VkBufferMemoryBarrier const barrier = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.pNext = nullptr,
		.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer = src->buffer,
		.offset = info.byteOffset,
		.size = info.dst.size(),
	};
	vkCmdPipelineBarrier(
		cmd,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, 0, nullptr, 1, &barrier, 0, nullptr
	);

	VkBufferCopy const region = {
		.srcOffset = info.byteOffset,
		.dstOffset = 0,
		.size = info.dst.size(),
	};
	vkCmdCopyBuffer(cmd, src->buffer, stagingBuffer, 1, &region);
	VkAssert(vkEndCommandBuffer(cmd));

	VkSubmitInfo const submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = nullptr,
		.waitSemaphoreCount = 0,
		.pWaitSemaphores = nullptr,
		.pWaitDstStageMask = nullptr,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd,
		.signalSemaphoreCount = 0,
		.pSignalSemaphores = nullptr,
	};
	VkAssert(
		vkQueueSubmit(sDevice->queueGraphics, 1, &submitInfo, VK_NULL_HANDLE)
	);
	VkAssert(vkQueueWaitIdle(sDevice->queueGraphics));

	vkFreeCommandBuffers(
		sDevice->device, sDevice->commandPoolGraphics, 1, &cmd
	);

	// -- copy staging -> caller buffer
	auto dst = info.dst;
	std::memcpy(dst.ptr(), stagingInfo.pMappedData, dst.size());

	vmaDestroyBuffer(sDevice->allocator, stagingBuffer, stagingAlloc);
}

// -----------------------------------------------------------------------------
// -- image
// -----------------------------------------------------------------------------

vkof::Image vkof::image_create(
	vkof::ImageCreateInfo const & createInfo,
	std::source_location loc
)
{
	ImplTexture implTexture;
	VkFormat const vkFormat = to_vk_format(createInfo.format);
	bool const is3d = createInfo.depth > 1u;
	VkImageCreateInfo const imageCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.imageType = is3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
		.format = vkFormat,
		.extent = VkExtent3D {
			.width = createInfo.width,
			.height = createInfo.height,
			.depth = is3d ? createInfo.depth : 1u,
		},
		.mipLevels = createInfo.mipLevels,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = format_usage(sDevice->physicalDevice, vkFormat),
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = nullptr,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	VmaAllocationCreateInfo const allocationCreateInfo = {
		.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
		.requiredFlags = 0,
		.preferredFlags = 0,
		.memoryTypeBits = 0,
		.pool = nullptr,
		.pUserData = nullptr,
		.priority = 0.0f,
	};
	VkAssert(
		vmaCreateImage(
			sDevice->allocator,
			&imageCreateInfo,
			&allocationCreateInfo,
			&implTexture.image,
			&implTexture.allocation,
			nullptr
		)
	);
	implTexture.width = createInfo.width;
	implTexture.height = createInfo.height;
	implTexture.depth = is3d ? createInfo.depth : 1u;

	VkImageViewType const viewType = (
		is3d ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D
	);

	VkImageViewCreateInfo const imageViewFullCi = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.image = implTexture.image,
		.viewType = viewType,
		.format = vkFormat,
		.components = VkComponentMapping {
			.r = VK_COMPONENT_SWIZZLE_IDENTITY,
			.g = VK_COMPONENT_SWIZZLE_IDENTITY,
			.b = VK_COMPONENT_SWIZZLE_IDENTITY,
			.a = VK_COMPONENT_SWIZZLE_IDENTITY,
		},
		.subresourceRange = VkImageSubresourceRange {
			.aspectMask = format_aspect(vkFormat),
			.baseMipLevel = 0,
			.levelCount = VK_REMAINING_MIP_LEVELS,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};
	VkAssert(
		vkCreateImageView(
			sDevice->device,
			&imageViewFullCi,
			nullptr,
			&implTexture.imageViewFull
		)
	);

	implTexture.imageViewPerMip.resize(createInfo.mipLevels);
	for (u32 mip = 0; mip < createInfo.mipLevels; ++mip) {
		VkImageViewCreateInfo const imageViewMipCi = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.image = implTexture.image,
			.viewType = viewType,
			.format = vkFormat,
			.components = VkComponentMapping {
				.r = VK_COMPONENT_SWIZZLE_IDENTITY,
				.g = VK_COMPONENT_SWIZZLE_IDENTITY,
				.b = VK_COMPONENT_SWIZZLE_IDENTITY,
				.a = VK_COMPONENT_SWIZZLE_IDENTITY,
			},
			.subresourceRange = VkImageSubresourceRange {
				.aspectMask = format_aspect(vkFormat),
				.baseMipLevel = mip,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		VkAssert(
			vkCreateImageView(
				sDevice->device,
				&imageViewMipCi,
				nullptr,
				&implTexture.imageViewPerMip[mip]
			)
		);
	}

	if (createInfo.optInitialData.size() > 0) {
		VkBufferCreateInfo const stagingCi = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.size = createInfo.optInitialData.size(),
			.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr,
		};
		VmaAllocationCreateInfo const stagingAllocCi = {
			.flags = (
				  VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
				| VMA_ALLOCATION_CREATE_MAPPED_BIT
				| VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
			),
			.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
			.requiredFlags = 0,
			.preferredFlags = 0,
			.memoryTypeBits = 0,
			.pool = nullptr,
			.pUserData = nullptr,
			.priority = 0.0f,
		};
		VkBuffer stagingBuffer;
		VmaAllocation stagingAlloc;
		VmaAllocationInfo stagingInfo;
		VkAssert(
			vmaCreateBuffer(
				sDevice->allocator,
				&stagingCi,
				&stagingAllocCi,
				&stagingBuffer,
				&stagingAlloc,
				&stagingInfo
			)
		);
		std::memcpy(
			stagingInfo.pMappedData,
			createInfo.optInitialData.ptr(),
			createInfo.optInitialData.size()
		);

		VkCommandBufferAllocateInfo const cbai = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.pNext = nullptr,
			.commandPool = sDevice->commandPoolGraphics,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		VkCommandBuffer cmd;
		VkAssert(vkAllocateCommandBuffers(sDevice->device, &cbai, &cmd));
		VkCommandBufferBeginInfo const beginInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.pNext = nullptr,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
			.pInheritanceInfo = nullptr,
		};
		VkAssert(vkBeginCommandBuffer(cmd, &beginInfo));

		VkImageMemoryBarrier toTransferDst = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.pNext = nullptr,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = implTexture.image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = VK_REMAINING_MIP_LEVELS,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0, 0, nullptr, 0, nullptr,
			1, &toTransferDst
		);

		VkBufferImageCopy const copyRegion = {
			.bufferOffset = 0,
			.bufferRowLength = 0,
			.bufferImageHeight = 0,
			.imageSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = 0,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
			.imageOffset = { 0, 0, 0 },
			.imageExtent = {
				createInfo.width,
				createInfo.height,
				is3d ? createInfo.depth : 1u,
			},
		};
		vkCmdCopyBufferToImage(
			cmd, stagingBuffer, implTexture.image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &copyRegion
		);

		VkImageMemoryBarrier const toGeneral = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.pNext = nullptr,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_GENERAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = implTexture.image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = VK_REMAINING_MIP_LEVELS,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			0, 0, nullptr, 0, nullptr,
			1, &toGeneral
		);

		VkAssert(vkEndCommandBuffer(cmd));
		VkSubmitInfo const submitInfo = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.pNext = nullptr,
			.waitSemaphoreCount = 0,
			.pWaitSemaphores = nullptr,
			.pWaitDstStageMask = nullptr,
			.commandBufferCount = 1,
			.pCommandBuffers = &cmd,
			.signalSemaphoreCount = 0,
			.pSignalSemaphores = nullptr,
		};
		VkAssert(
			vkQueueSubmit(sDevice->queueGraphics, 1, &submitInfo, VK_NULL_HANDLE)
		);
		VkAssert(vkQueueWaitIdle(sDevice->queueGraphics));
		vkFreeCommandBuffers(
			sDevice->device, sDevice->commandPoolGraphics, 1, &cmd
		);
		vmaDestroyBuffer(sDevice->allocator, stagingBuffer, stagingAlloc);
	} else {
		VkCommandBufferAllocateInfo const cbai = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.pNext = nullptr,
			.commandPool = sDevice->commandPoolGraphics,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		VkCommandBuffer cmd;
		VkAssert(vkAllocateCommandBuffers(sDevice->device, &cbai, &cmd));
		VkCommandBufferBeginInfo const beginInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.pNext = nullptr,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
			.pInheritanceInfo = nullptr,
		};
		VkAssert(vkBeginCommandBuffer(cmd, &beginInfo));
		VkImageMemoryBarrier const toGeneral = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.pNext = nullptr,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_GENERAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = implTexture.image,
			.subresourceRange = {
				.aspectMask = format_aspect(vkFormat),
				.baseMipLevel = 0,
				.levelCount = VK_REMAINING_MIP_LEVELS,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			0, 0, nullptr, 0, nullptr,
			1, &toGeneral
		);
		VkAssert(vkEndCommandBuffer(cmd));
		VkSubmitInfo const submitInfo = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.pNext = nullptr,
			.waitSemaphoreCount = 0,
			.pWaitSemaphores = nullptr,
			.pWaitDstStageMask = nullptr,
			.commandBufferCount = 1,
			.pCommandBuffers = &cmd,
			.signalSemaphoreCount = 0,
			.pSignalSemaphores = nullptr,
		};
		VkAssert(
			vkQueueSubmit(sDevice->queueGraphics, 1, &submitInfo, VK_NULL_HANDLE)
		);
		VkAssert(vkQueueWaitIdle(sDevice->queueGraphics));
		vkFreeCommandBuffers(
			sDevice->device, sDevice->commandPoolGraphics, 1, &cmd
		);
	}

	implTexture.vkFormat = vkFormat;

	vkof::Image const handle = sDevice->imagePool.allocate(implTexture);
	set_debug_name(
		VK_OBJECT_TYPE_IMAGE,
		(u64)implTexture.image,
		loc
	);
	return handle;
}

void vkof::image_destroy(Image const & image)
{
	auto const implTexture = sDevice->imagePool.get(image);
	if (implTexture) {
		vkDestroyImageView(
			sDevice->device, implTexture->imageViewFull, nullptr
		);
		for (u32 mip = 0; mip < implTexture->imageViewPerMip.size(); ++mip) {
			vkDestroyImageView(
				sDevice->device, implTexture->imageViewPerMip[mip], nullptr
			);
		}
		vmaDestroyImage(
			sDevice->allocator, implTexture->image, implTexture->allocation
		);
		sDevice->imagePool.free(image);
	}

	// release any bindless slots this image was ever registered under back
	// to their free lists, so a later image doesn't leave the descriptor
	// set growing forever with entries pointing at destroyed image views
	auto const slotsIt = sImageBindlessSlots.find(image.id);
	if (slotsIt != sImageBindlessSlots.end()) {
		for (auto const & [binding, slot] : slotsIt->second) {
			switch (binding) {
				case 0u:
					sBindlessSamplerFreeSlots.push_back(slot);
					break;
				case 1u:
					sBindlessStorageUintFreeSlots.push_back(slot);
					break;
				case 2u:
					sBindlessStorageFloatFreeSlots.push_back(slot);
					break;
				case 4u:
					sBindlessSampler3dFreeSlots.push_back(slot);
					break;
				case 5u:
					sBindlessStorage3dFreeSlots.push_back(slot);
					break;
				default:
					break;
			}
		}
		sImageBindlessSlots.erase(slotsIt);
	}
}

u32 vkof::image_width(Image const & image)
{
	auto const implTexture = sDevice->imagePool.get(image);
	return implTexture ? implTexture->width : 0;
}

u32 vkof::image_height(Image const & image)
{
	auto const implTexture = sDevice->imagePool.get(image);
	return implTexture ? implTexture->height : 0;
}

u32 vkof::image_depth(Image const & image)
{
	auto const implTexture = sDevice->imagePool.get(image);
	return implTexture ? implTexture->depth : 0;
}

void vkof::image_generate_mipmaps(Image const & image)
{
	auto const implTexture = sDevice->imagePool.get(image);
	if (!implTexture) { return; }
	u32 const mipLevels = (u32)implTexture->imageViewPerMip.size();
	if (mipLevels <= 1u) { return; }

	VkCommandBufferAllocateInfo const cbai = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = nullptr,
		.commandPool = sDevice->commandPoolGraphics,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer cmd;
	VkAssert(vkAllocateCommandBuffers(sDevice->device, &cbai, &cmd));
	VkCommandBufferBeginInfo const beginInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = nullptr,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = nullptr,
	};
	VkAssert(vkBeginCommandBuffer(cmd, &beginInfo));

	VkImageMemoryBarrier mip0ToSrc = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.pNext = nullptr,
		.srcAccessMask = 0,
		.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
		.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = implTexture->image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};
	vkCmdPipelineBarrier(
		cmd,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, 0, nullptr, 0, nullptr,
		1, &mip0ToSrc
	);

	i32 srcW = (i32)implTexture->width;
	i32 srcH = (i32)implTexture->height;

	for (u32 mip = 1u; mip < mipLevels; ++mip) {
		i32 const dstW = std::max(1, srcW / 2);
		i32 const dstH = std::max(1, srcH / 2);

		VkImageMemoryBarrier toDst = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.pNext = nullptr,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
			.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = implTexture->image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = mip,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0, 0, nullptr, 0, nullptr,
			1, &toDst
		);

		VkImageBlit const blit = {
			.srcSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = mip - 1u,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
			.srcOffsets = { { 0, 0, 0 }, { srcW, srcH, 1 } },
			.dstSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = mip,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
			.dstOffsets = { { 0, 0, 0 }, { dstW, dstH, 1 } },
		};
		vkCmdBlitImage(
			cmd,
			implTexture->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			implTexture->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &blit,
			VK_FILTER_LINEAR
		);

		VkImageMemoryBarrier dstToSrc = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.pNext = nullptr,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = implTexture->image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = mip,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		vkCmdPipelineBarrier(
			cmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0, 0, nullptr, 0, nullptr,
			1, &dstToSrc
		);

		srcW = dstW;
		srcH = dstH;
	}

	VkImageMemoryBarrier toShaderRead = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.pNext = nullptr,
		.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
		.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		.newLayout = VK_IMAGE_LAYOUT_GENERAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = implTexture->image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = VK_REMAINING_MIP_LEVELS,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};
	vkCmdPipelineBarrier(
		cmd,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0, 0, nullptr, 0, nullptr,
		1, &toShaderRead
	);

	VkAssert(vkEndCommandBuffer(cmd));
	VkSubmitInfo const submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = nullptr,
		.waitSemaphoreCount = 0,
		.pWaitSemaphores = nullptr,
		.pWaitDstStageMask = nullptr,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd,
		.signalSemaphoreCount = 0,
		.pSignalSemaphores = nullptr,
	};
	VkAssert(
		vkQueueSubmit(sDevice->queueGraphics, 1, &submitInfo, VK_NULL_HANDLE)
	);
	VkAssert(vkQueueWaitIdle(sDevice->queueGraphics));
	vkFreeCommandBuffers(
		sDevice->device, sDevice->commandPoolGraphics, 1, &cmd
	);
}

vkof::Sampler vkof::sampler_create(SamplerCreateInfo const & createInfo)
{
	auto const toFilter = [](SamplerFilter f) {
		return (
			f == SamplerFilter::nearest
				? VK_FILTER_NEAREST
				: VK_FILTER_LINEAR
		);
	};
	auto const toAddress = [](SamplerAddressMode m) {
		switch (m) {
			case SamplerAddressMode::repeat:
				return VK_SAMPLER_ADDRESS_MODE_REPEAT;
			case SamplerAddressMode::mirrored_repeat:
				return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
			default:
				return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		}
	};

	ImplSampler implSampler;
	VkSamplerCreateInfo const samplerCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.magFilter = toFilter(createInfo.magFilter),
		.minFilter = toFilter(createInfo.minFilter),
		.mipmapMode = (
			createInfo.mipmapMode == SamplerMipmapMode::nearest
			? VK_SAMPLER_MIPMAP_MODE_NEAREST
			: VK_SAMPLER_MIPMAP_MODE_LINEAR
		),
		.addressModeU = toAddress(createInfo.addressModeU),
		.addressModeV = toAddress(createInfo.addressModeV),
		.addressModeW = toAddress(createInfo.addressModeW),
		.mipLodBias = 0.0f,
		.anisotropyEnable = createInfo.maxAnisotropy > 0.0f ? VK_TRUE : VK_FALSE,
		.maxAnisotropy = createInfo.maxAnisotropy,
		.compareEnable = VK_FALSE,
		.compareOp = VK_COMPARE_OP_ALWAYS,
		.minLod = 0.0f,
		.maxLod = VK_LOD_CLAMP_NONE,
		.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
		.unnormalizedCoordinates = VK_FALSE,
	};
	VkAssert(
		vkCreateSampler(
			sDevice->device,
			&samplerCreateInfo,
			nullptr,
			&implSampler.sampler
		)
	);
	return sDevice->samplerPool.allocate(implSampler);
}

void vkof::sampler_destroy(Sampler const & sampler)
{
	auto const implSampler = sDevice->samplerPool.get(sampler);
	if (implSampler) {
		vkDestroySampler(sDevice->device, implSampler->sampler, nullptr);
		sDevice->samplerPool.free(sampler);
	}
}

u32 vkof::image_sampler_handle(ImageSamplerHandleInfo const & info)
{
	auto const implTexture = sDevice->imagePool.get(info.image);
	auto const implSampler = sDevice->samplerPool.get(info.sampler);

	if (!implTexture || !implSampler) {
		return 0;
	}

	u32 const slot = (
		alloc_bindless_slot(sBindlessSamplerFreeSlots, sBindlessSamplerNextSlot)
	);
	VkDescriptorImageInfo const imageInfo = {
		.sampler = implSampler->sampler,
		.imageView = implTexture->imageViewFull,
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
	};
	VkWriteDescriptorSet const write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.pNext = nullptr,
		.dstSet = sBindlessSet,
		.dstBinding = 0u,
		.dstArrayElement = slot,
		.descriptorCount = 1u,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &imageInfo,
		.pBufferInfo = nullptr,
		.pTexelBufferView = nullptr,
	};
	vkUpdateDescriptorSets(sDevice->device, 1u, &write, 0u, nullptr);
	sImageBindlessSlots[info.image.id].emplace_back(0u, slot);
	return slot;
}

ImTextureID vkof::image_imgui_id(ImageSamplerHandleInfo const & info)
{
	auto const implTexture = sDevice->imagePool.get(info.image);
	auto const implSampler = sDevice->samplerPool.get(info.sampler);
	if (!implTexture || !implSampler) { return ImTextureID(0); }
	VkDescriptorSet const set = ImGui_ImplVulkan_AddTexture(
		implSampler->sampler,
		implTexture->imageViewFull,
		VK_IMAGE_LAYOUT_GENERAL
	);
	return reinterpret_cast<ImTextureID>(set);
}

void vkof::image_imgui_id_destroy(ImTextureID const id)
{
	if (!id) { return; }
	ImGui_ImplVulkan_RemoveTexture(reinterpret_cast<VkDescriptorSet>(id));
}

static VkImageView image_storage_view_for_mip(
	ImplTexture const & implTexture, u32 mipLevel
)
{
	if (mipLevel >= implTexture.imageViewPerMip.size()) {
		return VK_NULL_HANDLE;
	}
	return implTexture.imageViewPerMip[mipLevel];
}

u32 vkof::image_storage_handle(ImageStorageHandleInfo const & info)
{
	auto const implTexture = sDevice->imagePool.get(info.image);
	if (!implTexture) {
		return 0;
	}

	VkImageView const view = (
		info.mipLevel == 0
		? implTexture->imageViewFull
		: image_storage_view_for_mip(*implTexture, info.mipLevel)
	);

	bool const isUint = format_is_uint(implTexture->vkFormat);
	u32 const binding = isUint ? 1u : 2u;
	std::vector<u32> & freeSlots = (
		isUint ? sBindlessStorageUintFreeSlots : sBindlessStorageFloatFreeSlots
	);
	u32 & slotCounter = (
		isUint ? sBindlessStorageUintNextSlot : sBindlessStorageFloatNextSlot
	);
	u32 const slot = alloc_bindless_slot(freeSlots, slotCounter);
	VkDescriptorImageInfo const imageInfo = {
		.sampler = VK_NULL_HANDLE,
		.imageView = view,
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
	};
	VkWriteDescriptorSet const write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.pNext = nullptr,
		.dstSet = sBindlessSet,
		.dstBinding = binding,
		.dstArrayElement = slot,
		.descriptorCount = 1u,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.pImageInfo = &imageInfo,
		.pBufferInfo = nullptr,
		.pTexelBufferView = nullptr,
	};
	vkUpdateDescriptorSets(sDevice->device, 1u, &write, 0u, nullptr);
	sImageBindlessSlots[info.image.id].emplace_back(binding, slot);
	return slot;
}

u32 vkof::image_sampler3d_handle(ImageSamplerHandleInfo const & info)
{
	auto const implTexture = sDevice->imagePool.get(info.image);
	auto const implSampler = sDevice->samplerPool.get(info.sampler);
	if (!implTexture || !implSampler) { return 0u; }

	u32 const slot = (
		alloc_bindless_slot(
			sBindlessSampler3dFreeSlots, sBindlessSampler3dNextSlot
		)
	);
	VkDescriptorImageInfo const imageInfo = {
		.sampler = implSampler->sampler,
		.imageView = implTexture->imageViewFull,
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
	};
	VkWriteDescriptorSet const write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.pNext = nullptr,
		.dstSet = sBindlessSet,
		.dstBinding = 4u,
		.dstArrayElement = slot,
		.descriptorCount = 1u,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &imageInfo,
		.pBufferInfo = nullptr,
		.pTexelBufferView = nullptr,
	};
	vkUpdateDescriptorSets(sDevice->device, 1u, &write, 0u, nullptr);
	sImageBindlessSlots[info.image.id].emplace_back(4u, slot);
	return slot;
}

u32 vkof::image_storage3d_handle(ImageStorageHandleInfo const & info)
{
	auto const implTexture = sDevice->imagePool.get(info.image);
	if (!implTexture) { return 0u; }

	VkImageView const view = (
		info.mipLevel == 0u
		? implTexture->imageViewFull
		: image_storage_view_for_mip(*implTexture, info.mipLevel)
	);

	u32 const slot = (
		alloc_bindless_slot(
			sBindlessStorage3dFreeSlots, sBindlessStorage3dNextSlot
		)
	);
	VkDescriptorImageInfo const imageInfo = {
		.sampler = VK_NULL_HANDLE,
		.imageView = view,
		.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
	};
	VkWriteDescriptorSet const write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.pNext = nullptr,
		.dstSet = sBindlessSet,
		.dstBinding = 5u,
		.dstArrayElement = slot,
		.descriptorCount = 1u,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.pImageInfo = &imageInfo,
		.pBufferInfo = nullptr,
		.pTexelBufferView = nullptr,
	};
	vkUpdateDescriptorSets(sDevice->device, 1u, &write, 0u, nullptr);
	sImageBindlessSlots[info.image.id].emplace_back(5u, slot);
	return slot;
}

u32 vkof::transient_image_storage_handle(
	TransientImageStorageHandleInfo const & info
) {
	auto const impl = sDevice->transientImagePool.get(info.image);
	if (!impl) { return 0u; }

	u32 const pingPongSlot = (
		impl->isDoubleBuffered ? (sDevice->frameIndex % 2u) : 0u
	);
	vkof::Image const img = impl->handles[pingPongSlot];
	auto const implTexture = sDevice->imagePool.get(img);
	if (!implTexture) { return 0u; }

	u64 const cacheKey = (
		transient_storage_cache_key(info.image.id, info.mipLevel, pingPongSlot)
	);
	auto const it = sTransientStorageSlotCache.find(cacheKey);
	if (it != sTransientStorageSlotCache.end()) {
		// already written for this exact physical image; rewriting here
		// would race a still-in-flight frame reading through this same
		// slot via the *other* ping-pong image (see kFramesInFlight)
		return it->second;
	}

	bool const isUint = format_is_uint(implTexture->vkFormat);
	u32 const binding = isUint ? 1u : 2u;
	std::vector<u32> & freeSlots = (
		isUint ? sBindlessStorageUintFreeSlots : sBindlessStorageFloatFreeSlots
	);
	u32 & slotCounter = (
		isUint ? sBindlessStorageUintNextSlot : sBindlessStorageFloatNextSlot
	);
	u32 const slot = alloc_bindless_slot(freeSlots, slotCounter);
	sTransientStorageSlotCache.emplace(cacheKey, slot);
	write_storage_image_descriptor(*implTexture, info.mipLevel, binding, slot);
	return slot;
}

// -----------------------------------------------------------------------------
// -- transient resources
// -----------------------------------------------------------------------------

vkof::TransientImage vkof::transient_image_create(
	TransientImageCreateInfo const & createInfo
) {
	auto const w = (u32)(
		sDevice->renderExtent.width * createInfo.scaleWidth
	);
	auto const h = (u32)(
		sDevice->renderExtent.height * createInfo.scaleHeight
	);

	ImplTransientImage impl = {
		.format = createInfo.format,
		.scaleWidth = createInfo.scaleWidth,
		.scaleHeight = createInfo.scaleHeight,
		.mipLevels = createInfo.mipLevels,
		.isDoubleBuffered = createInfo.isDoubleBuffered,
		.handles = {
			vkof::Image { .id = 0 },
			vkof::Image { .id = 0 },
		},
	};

	auto const allocSlot = [&](u32 slot) {
		vkof::ImageCreateInfo const ci {
			.width = w,
			.height = h,
			.format = createInfo.format,
			.mipLevels = createInfo.mipLevels,
			.optInitialData = {},
		};
		impl.handles[slot] = vkof::image_create(ci);
	};

	allocSlot(0);
	if (createInfo.isDoubleBuffered) {
		allocSlot(1);
	}

	vkof::TransientImage const handle = (
		sDevice->transientImagePool.allocate(impl)
	);
	sDevice->transientImageHandles.emplace_back(handle);
	return handle;
}

vkof::TransientBuffer vkof::transient_buffer_create(
	TransientBufferCreateInfo const & createInfo
) {
	ImplTransientBuffer impl;
	impl.isDoubleBuffered = createInfo.isDoubleBuffered;
	impl.handles[0] = { .id = 0 };
	impl.handles[1] = { .id = 0 };

	auto const allocSlot = [&](u32 slot) {
		vkof::BufferCreateInfo const ci {
			.byteCount = createInfo.byteCount,
			.memory    = vkof::BufferMemory::DeviceOnly,
		};
		impl.handles[slot] = vkof::buffer_create(ci);
		sDevice->transientBufferCleanupHandles.push_back(impl.handles[slot]);
	};

	allocSlot(0);
	if (createInfo.isDoubleBuffered) {
		allocSlot(1);
	}

	return sDevice->transientBufferPool.allocate(impl);
}

void vkof::render_extent_set(u32 const width, u32 const height)
{
	if (
		width == sDevice->renderExtent.width
		&& height == sDevice->renderExtent.height
	) {
		return;
	}

	vkof::device_wait_idle();
	sDevice->renderExtent = VkExtent2D { width, height };

	// recreate the backing images of every live transient image in place;
	//   the TransientImage handles held by the caller stay valid
	for (vkof::TransientImage const & ti : sDevice->transientImageHandles) {
		auto const impl = sDevice->transientImagePool.get(ti);
		if (!impl) { continue; }
		auto const w = (u32)(width * impl->scaleWidth);
		auto const h = (u32)(height * impl->scaleHeight);
		for (u32 slot = 0u; slot < 2u; ++slot) {
			if (impl->handles[slot].id == 0u) { continue; }
			vkof::image_destroy(impl->handles[slot]);
			vkof::ImageCreateInfo const ci {
				.width = w,
				.height = h,
				.format = impl->format,
				.mipLevels = impl->mipLevels,
				.optInitialData = {},
			};
			impl->handles[slot] = vkof::image_create(ci);

			// any bindless storage slot cached for this transient image's
			// this ping-pong physical image now points at the view just
			// destroyed above; repoint it (same slot number) at the fresh
			// image instead of leaving a dangling descriptor entry
			auto const implTexture = (
				sDevice->imagePool.get(impl->handles[slot])
			);
			if (!implTexture) { continue; }
			bool const isUint = format_is_uint(implTexture->vkFormat);
			u32 const binding = isUint ? 1u : 2u;
			for (auto const & [cacheKey, cachedSlot] : sTransientStorageSlotCache) {
				bool const sameImage = (cacheKey >> 16u) == ti.id;
				bool const samePingPongSlot = u32(cacheKey & 0x1u) == slot;
				if (!sameImage || !samePingPongSlot) { continue; }
				u32 const mipLevel = u32((cacheKey >> 1u) & 0x7fffu);
				write_storage_image_descriptor(
					*implTexture, mipLevel, binding, cachedSlot
				);
			}
		}
	}
}

vkof::Image vkof::transient_image_get_image(
	TransientImage const & transientImage
) {
	auto const impl = sDevice->transientImagePool.get(transientImage);
	if (!impl) {
		return vkof::Image { .id = 0 };
	}
	u32 const slot = (
		impl->isDoubleBuffered
			? (sDevice->frameIndex % 2)
			: 0u
	);
	return impl->handles[slot];
}

vkof::Buffer vkof::transient_buffer_get_buffer(
	TransientBuffer const & transientBuffer
) {
	auto const impl = sDevice->transientBufferPool.get(transientBuffer);
	if (!impl) {
		return vkof::Buffer { .id = 0 };
	}
	u32 const slot = (
		impl->isDoubleBuffered
			? (sDevice->frameIndex % 2)
			: 0u
	);
	return impl->handles[slot];
}

// -----------------------------------------------------------------------------
// -- render nodes
// -----------------------------------------------------------------------------

vkof::RenderNode vkof::render_node_create(
	RenderNodeCreateInfo const & createInfo
) {
	ImplRenderNode impl;
	impl.queue = createInfo.queue;
	return sDevice->renderNodePool.allocate(impl);
}

void vkof::render_node_destroy(RenderNode const & node)
{
	sDevice->renderNodePool.free(node);
}

void vkof::render_node_add_image(RenderNodeImageInfo const & info)
{
	auto const node = sDevice->renderNodePool.get(info.node);
	if (!node) { return; }
	node->imageBarriers.push_back(
		ImplRenderBarrier {
			.transientImage = info.image,
			.access = info.access,
		}
	);
}

void vkof::render_node_add_persistent_image(
	RenderNodePersistentImageInfo const & info
) {
	auto const node = sDevice->renderNodePool.get(info.node);
	if (!node) { return; }
	node->persistentImages.emplace_back(info.image);
}

void vkof::render_node_add_buffer(RenderNodeBufferInfo const & info)
{
	auto const node = sDevice->renderNodePool.get(info.node);
	if (!node) { return; }
	node->bufferBarriers.push_back(
		ImplBufferBarrier {
			.transientBuffer = info.buffer,
			.access = info.access,
		}
	);
}

void vkof::render_node_attachment_color(
	RenderNodeAttachmentColorInfo const & info
) {
	auto const node = sDevice->renderNodePool.get(info.node);
	if (!node) { return; }
	ImplColorAttachment att;
	att.image     = info.image;
	att.loadOp    = info.loadOp;
	att.mipLevel  = info.mipLevel;
	att.colorIndex = info.colorIndex;
	att.clearColor[0] = (
		(info.clearColor.size() > 0) ? info.clearColor.ptr()[0] : 0.0f
	);
	att.clearColor[1] = (
		(info.clearColor.size() > 1) ? info.clearColor.ptr()[1] : 0.0f
	);
	att.clearColor[2] = (
		(info.clearColor.size() > 2) ? info.clearColor.ptr()[2] : 0.0f
	);
	att.clearColor[3] = (
		(info.clearColor.size() > 3) ? info.clearColor.ptr()[3] : 1.0f
	);
	node->colorAttachments.push_back(att);
}

void vkof::render_node_attachment_depth(
	RenderNodeAttachmentDepthInfo const & info
) {
	auto const node = sDevice->renderNodePool.get(info.node);
	if (!node) { return; }
	node->depthAttachment = ImplDepthAttachment {
		.image      = info.image,
		.loadOp     = info.loadOp,
		.mipLevel   = info.mipLevel,
		.clearDepth = (
			(info.clearDepth.size() > 0) ? info.clearDepth.ptr()[0] : 1.0f
		),
	};
}

void vkof::render_node_callback(RenderNodeCallbackInfo const & info)
{
	auto const node = sDevice->renderNodePool.get(info.node);
	if (!node) { return; }
	node->callback = info.callback;
}

// -----------------------------------------------------------------------------
// -- creation/destruction
// -----------------------------------------------------------------------------

static void init_impl(
	bool const isHeadless, u32 const & width, u32 const & height
)
{
#if defined(VKOF_AFTERMATH)
	vkof_aftermath_enable();
#endif

	// -- window (windowed only)
	GLFWwindow * window = nullptr;
	if (!isHeadless) {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
		window = (
			glfwCreateWindow(
				(i32)width, (i32)height, "vkof", nullptr, nullptr
			)
		);
	}

	// -- instance
	auto instance = [&]() -> vkb::Instance {
		vkb::InstanceBuilder builder;
		builder
			.set_app_name(isHeadless ? "vkof-test" : "demo")
			.request_validation_layers(true)
			.set_debug_callback(debugMessengerCallback)
			.add_debug_messenger_severity(
				VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
			)
			.add_validation_feature_enable(
				VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT
			)
			.require_api_version(1, 3, 0);
		if (isHeadless) {
			builder.set_headless(true);
		}
		auto const r = builder.build();
		if (!r) {
			printf(
				"failed to create instance: %s\n",
				r.error().message().c_str()
			);
			exit(1);
		}
		return r.value();
	}();

	// -- surface (windowed only)
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	if (!isHeadless) {
		VkAssert(
			glfwCreateWindowSurface(
				instance.instance,
				window,
				/*allocator=*/ nullptr,
				&surface
			)
		);
	}

	// -- device
	auto device = [&]() -> vkb::Device {
		VkPhysicalDeviceFeatures features {
			.robustBufferAccess = true,
			.fullDrawIndexUint32 = true,
			.imageCubeArray = true,
			.independentBlend = true,
			.geometryShader = true,
			.tessellationShader = true,
			.sampleRateShading = true,
			.dualSrcBlend = false,
			.logicOp = true,
			.multiDrawIndirect = true,
			.drawIndirectFirstInstance = true,
			.depthClamp = true,
			.depthBiasClamp = true,
			.fillModeNonSolid = true,
			.depthBounds = true,
			.wideLines = false,
			.largePoints = false,
			.alphaToOne = true,
			.multiViewport = true,
			.samplerAnisotropy = true,
			.textureCompressionETC2 = false,
			.textureCompressionASTC_LDR = false,
			.textureCompressionBC = false,
			.occlusionQueryPrecise = false,
			.pipelineStatisticsQuery = false,
			.vertexPipelineStoresAndAtomics = true,
			.fragmentStoresAndAtomics = true,
			.shaderTessellationAndGeometryPointSize = false,
			.shaderImageGatherExtended = false,
			.shaderStorageImageExtendedFormats = true,
			.shaderStorageImageMultisample = true,
			.shaderStorageImageReadWithoutFormat = true,
			.shaderStorageImageWriteWithoutFormat = true,
			.shaderUniformBufferArrayDynamicIndexing = true,
			.shaderSampledImageArrayDynamicIndexing = true,
			.shaderStorageBufferArrayDynamicIndexing = true,
			.shaderStorageImageArrayDynamicIndexing = true,
			.shaderClipDistance = false,
			.shaderCullDistance = false,
			.shaderFloat64 = true,
			.shaderInt64 = true,
			.shaderInt16 = true,
			.shaderResourceResidency = false,
			.shaderResourceMinLod = false,
			.sparseBinding = false,
			.sparseResidencyBuffer = false,
			.sparseResidencyImage2D = false,
			.sparseResidencyImage3D = false,
			.sparseResidency2Samples = false,
			.sparseResidency4Samples = false,
			.sparseResidency8Samples = false,
			.sparseResidency16Samples = false,
			.sparseResidencyAliased = false,
			.variableMultisampleRate = false,
			.inheritedQueries = false,
		};
		VkPhysicalDeviceMeshShaderFeaturesEXT const meshShaderFeatures = {
			.sType = (
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT
			),
			.pNext = nullptr,
			.taskShader = VK_TRUE,
			.meshShader = VK_TRUE,
			.multiviewMeshShader = VK_FALSE,
			.primitiveFragmentShadingRateMeshShader = VK_FALSE,
			.meshShaderQueries = VK_FALSE,
		};
		VkPhysicalDeviceAccelerationStructureFeaturesKHR const
			accelerationStructureFeatures =
		{
			.sType = (
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR
			),
			.pNext = nullptr,
			.accelerationStructure = VK_TRUE,
			.accelerationStructureCaptureReplay = VK_FALSE,
			.accelerationStructureIndirectBuild = VK_FALSE,
			.accelerationStructureHostCommands = VK_FALSE,
			.descriptorBindingAccelerationStructureUpdateAfterBind = VK_TRUE,
		};
		VkPhysicalDeviceShaderRelaxedExtendedInstructionFeaturesKHR const
			relaxedExtInstFeatures =
		{
			.sType = (
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_RELAXED_EXTENDED_INSTRUCTION_FEATURES_KHR
			),
			.pNext = nullptr,
			.shaderRelaxedExtendedInstruction = VK_TRUE,
		};
		VkPhysicalDeviceRayQueryFeaturesKHR const rayQueryFeatures = {
			.sType = (
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR
			),
			.pNext = nullptr,
			.rayQuery = VK_TRUE,
		};
		VkPhysicalDeviceMaintenance4Features const maintenance4Features = {
			.sType = (
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES
			),
			.pNext = nullptr,
			.maintenance4 = VK_TRUE,
		};
		VkPhysicalDeviceDynamicRenderingFeatures const
			dynamicRenderingFeatures = {
				.sType = (
					VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES
				),
				.pNext = nullptr,
				.dynamicRendering = VK_TRUE,
			};
		VkPhysicalDeviceSynchronization2Features const
			synchronization2Features = {
				.sType = (
					VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES
				),
				.pNext = nullptr,
				.synchronization2 = VK_TRUE,
			};
		VkPhysicalDeviceVulkan12Features const vulkan12Features {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
			.pNext = nullptr,
			.samplerMirrorClampToEdge = VK_TRUE,
			.drawIndirectCount = VK_TRUE,
			.storageBuffer8BitAccess = VK_TRUE,
			.uniformAndStorageBuffer8BitAccess = VK_TRUE,
			.storagePushConstant8 = VK_TRUE,
			.shaderBufferInt64Atomics = VK_TRUE,
			.shaderSharedInt64Atomics = VK_TRUE,
			.shaderFloat16 = VK_TRUE,
			.shaderInt8 = VK_TRUE,
			.descriptorIndexing = VK_TRUE,
			.shaderInputAttachmentArrayDynamicIndexing = VK_TRUE,
			.shaderUniformTexelBufferArrayDynamicIndexing = VK_TRUE,
			.shaderStorageTexelBufferArrayDynamicIndexing = VK_TRUE,
			.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE,
			.shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
			.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE,
			.shaderStorageImageArrayNonUniformIndexing = VK_TRUE,
			.shaderInputAttachmentArrayNonUniformIndexing = VK_TRUE,
			.shaderUniformTexelBufferArrayNonUniformIndexing = VK_TRUE,
			.shaderStorageTexelBufferArrayNonUniformIndexing = VK_TRUE,
			.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE,
			.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
			.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE,
			.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE,
			.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE,
			.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE,
			.descriptorBindingUpdateUnusedWhilePending = VK_TRUE,
			.descriptorBindingPartiallyBound = VK_TRUE,
			.descriptorBindingVariableDescriptorCount = VK_TRUE,
			.runtimeDescriptorArray = VK_TRUE,
			.samplerFilterMinmax = VK_TRUE,
			.scalarBlockLayout = VK_TRUE,
			.imagelessFramebuffer = VK_TRUE,
			.uniformBufferStandardLayout = VK_TRUE,
			.shaderSubgroupExtendedTypes = VK_TRUE,
			.separateDepthStencilLayouts = VK_TRUE,
			.hostQueryReset = VK_TRUE,
			.timelineSemaphore = VK_TRUE,
			.bufferDeviceAddress = VK_TRUE,
			.bufferDeviceAddressCaptureReplay = VK_TRUE,
			.bufferDeviceAddressMultiDevice = VK_TRUE,
			.vulkanMemoryModel = VK_TRUE,
			.vulkanMemoryModelDeviceScope = VK_TRUE,
			.vulkanMemoryModelAvailabilityVisibilityChains = VK_TRUE,
			.shaderOutputViewportIndex = VK_TRUE,
			.shaderOutputLayer = VK_TRUE,
			.subgroupBroadcastDynamicId = VK_TRUE,
		};
		vkb::PhysicalDeviceSelector selector(instance);
		if (isHeadless) {
			selector.defer_surface_initialization();
			selector.require_present(false);
		} else {
			selector.set_surface(surface);
		}
		auto const physResult = (
			selector
			.add_required_extension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME)
			.add_required_extension(VK_EXT_MESH_SHADER_EXTENSION_NAME)
			.set_required_features(features)
			.add_required_extension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)
			.add_required_extension(VK_KHR_MAINTENANCE_6_EXTENSION_NAME)
			.add_required_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
			.add_required_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
			.add_required_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME)
			.add_required_extension_features(meshShaderFeatures)
			.add_required_extension_features(accelerationStructureFeatures)
			.add_required_extension_features(rayQueryFeatures)
			.add_required_extension_features(maintenance4Features)
			.add_required_extension_features(dynamicRenderingFeatures)
			.add_required_extension_features(synchronization2Features)
			.add_required_extension_features(vulkan12Features)
#if defined(VKOF_AFTERMATH)
			.add_required_extension(
				VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME
			)
			.add_required_extension(
				VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME
			)
#endif
			.select()
		);
		if (!physResult) {
			printf(
				"failed to select physical device: %s\n",
				physResult.error().message().c_str()
			);
			exit(1);
		}
		vkb::DeviceBuilder devBuilder(physResult.value());
#if defined(VKOF_AFTERMATH)
		VkDeviceDiagnosticsConfigCreateInfoNV const diagnosticsConfigCi = {
			.sType = (
				VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV
			),
			.pNext = nullptr,
			.flags = (
				VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV |
				VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV |
				VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV |
				VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_ERROR_REPORTING_BIT_NV
			),
		};
		devBuilder.add_pNext(
			const_cast<VkDeviceDiagnosticsConfigCreateInfoNV *>(
				&diagnosticsConfigCi
			)
		);
#endif
		auto const devResult = devBuilder.build();
		if (!devResult) {
			printf(
				"failed to create logical device: %s\n",
				devResult.error().message().c_str()
			);
			exit(1);
		}
		return devResult.value();
	}();

	// -- queues
	auto const graphicsQueue = [&]() -> VkQueue {
		auto const r = device.get_queue(vkb::QueueType::graphics);
		if (!r) {
			printf(
				"failed to get graphics queue: %s\n",
				r.error().message().c_str()
			);
			exit(1);
		}
		return r.value();
	}();
	auto const computeQueue = [&]() -> VkQueue {
		auto const r = device.get_queue(vkb::QueueType::compute);
		if (!r) {
			printf(
				"failed to get compute queue: %s\n",
				r.error().message().c_str()
			);
			exit(1);
		}
		return r.value();
	}();

	// -- vma allocator
	auto const allocator = [&]() -> VmaAllocator {
		VmaAllocatorCreateInfo const allocatorCi = {
			.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
			.physicalDevice = device.physical_device,
			.device = device.device,
			.preferredLargeHeapBlockSize = 0,
			.pAllocationCallbacks = nullptr,
			.pDeviceMemoryCallbacks = nullptr,
			.pHeapSizeLimit = nullptr,
			.pVulkanFunctions = nullptr,
			.instance = instance.instance,
			.vulkanApiVersion = VK_API_VERSION_1_3,
			.pTypeExternalMemoryHandleTypes = nullptr,
		};
		VmaAllocator a;
		VkAssert(vmaCreateAllocator(&allocatorCi, &a));
		return a;
	}();

	// -- swapchain + depth images
	vkb::Swapchain swapchain;
	std::vector<VkImage> swapchainImages;
	std::vector<VkImageView> swapchainImageViews;
	std::vector<VkImage> depthImages;
	std::vector<VkImageView> depthImageViews;
	std::vector<VmaAllocation> depthAllocs;
	VmaAllocation headlessColorAlloc = {};
	VmaAllocation headlessDepthAlloc = {};

	if (isHeadless) {
		swapchain.device = device.device;
		swapchain.swapchain = VK_NULL_HANDLE;
		swapchain.image_count = 1;
		swapchain.image_format = VK_FORMAT_B8G8R8A8_UNORM;
		swapchain.extent = VkExtent2D { width, height };

		VkImage colorImage;
		VkImageView colorView;
		{
			VkImageCreateInfo const imageCi = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
				.pNext = nullptr,
				.flags = 0,
				.imageType = VK_IMAGE_TYPE_2D,
				.format = VK_FORMAT_B8G8R8A8_UNORM,
				.extent = VkExtent3D { width, height, 1 },
				.mipLevels = 1,
				.arrayLayers = 1,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.tiling = VK_IMAGE_TILING_OPTIMAL,
				.usage = (
					  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
					| VK_IMAGE_USAGE_SAMPLED_BIT
					| VK_IMAGE_USAGE_TRANSFER_SRC_BIT
					| VK_IMAGE_USAGE_TRANSFER_DST_BIT
					| VK_IMAGE_USAGE_STORAGE_BIT
				),
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
				.queueFamilyIndexCount = 0,
				.pQueueFamilyIndices = nullptr,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			};
			VmaAllocationCreateInfo const allocCi = {
				.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
				.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
				.requiredFlags = 0,
				.preferredFlags = 0,
				.memoryTypeBits = 0,
				.pool = nullptr,
				.pUserData = nullptr,
				.priority = 0.0f,
			};
			VkAssert(
				vmaCreateImage(
					allocator,
					&imageCi,
					&allocCi,
					&colorImage,
					&headlessColorAlloc,
					nullptr
				)
			);
			VkImageViewCreateInfo const viewCi = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.pNext = nullptr,
				.flags = 0,
				.image = colorImage,
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = VK_FORMAT_B8G8R8A8_UNORM,
				.components = VkComponentMapping {
					.r = VK_COMPONENT_SWIZZLE_IDENTITY,
					.g = VK_COMPONENT_SWIZZLE_IDENTITY,
					.b = VK_COMPONENT_SWIZZLE_IDENTITY,
					.a = VK_COMPONENT_SWIZZLE_IDENTITY,
				},
				.subresourceRange = VkImageSubresourceRange {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};
			VkAssert(
				vkCreateImageView(device.device, &viewCi, nullptr, &colorView)
			);
		}
		swapchainImages = { colorImage };
		swapchainImageViews = { colorView };

		VkImage depthImage;
		VkImageView depthView;
		{
			VkImageCreateInfo const imageCi = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
				.pNext = nullptr,
				.flags = 0,
				.imageType = VK_IMAGE_TYPE_2D,
				.format = VK_FORMAT_D32_SFLOAT,
				.extent = VkExtent3D { width, height, 1 },
				.mipLevels = 1,
				.arrayLayers = 1,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.tiling = VK_IMAGE_TILING_OPTIMAL,
				.usage = (
					VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
					| VK_IMAGE_USAGE_SAMPLED_BIT
				),
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
				.queueFamilyIndexCount = 0,
				.pQueueFamilyIndices = nullptr,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			};
			VmaAllocationCreateInfo const allocCi = {
				.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
				.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
				.requiredFlags = 0,
				.preferredFlags = 0,
				.memoryTypeBits = 0,
				.pool = nullptr,
				.pUserData = nullptr,
				.priority = 0.0f,
			};
			VkAssert(
				vmaCreateImage(
					allocator,
					&imageCi,
					&allocCi,
					&depthImage,
					&headlessDepthAlloc,
					nullptr
				)
			);
			VkImageViewCreateInfo const viewCi = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.pNext = nullptr,
				.flags = 0,
				.image = depthImage,
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = VK_FORMAT_D32_SFLOAT,
				.components = VkComponentMapping {
					.r = VK_COMPONENT_SWIZZLE_IDENTITY,
					.g = VK_COMPONENT_SWIZZLE_IDENTITY,
					.b = VK_COMPONENT_SWIZZLE_IDENTITY,
					.a = VK_COMPONENT_SWIZZLE_IDENTITY,
				},
				.subresourceRange = VkImageSubresourceRange {
					.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};
			VkAssert(
				vkCreateImageView(device.device, &viewCi, nullptr, &depthView)
			);
		}
		depthImages = { depthImage };
		depthImageViews = { depthView };
	} else {
		swapchain = [&]() -> vkb::Swapchain {
			auto const r = (
				vkb::SwapchainBuilder(device)
				.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
				.set_desired_extent(width, height)
				.add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
				.build()
			);
			if (!r) {
				printf(
					"failed to create swapchain: %s\n",
					r.error().message().c_str()
				);
				exit(1);
			}
			return r.value();
		}();
		swapchainImages = swapchain.get_images().value();
		swapchainImageViews = swapchain.get_image_views().value();

		depthAllocs.resize(swapchain.image_count);
		depthImages.resize(swapchain.image_count);
		depthImageViews.resize(swapchain.image_count);
		for (size_t i = 0; i < depthImages.size(); ++i) {
			VkImageCreateInfo const imageCi = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
				.pNext = nullptr,
				.flags = 0,
				.imageType = VK_IMAGE_TYPE_2D,
				.format = VK_FORMAT_D32_SFLOAT,
				.extent = VkExtent3D {
					.width = swapchain.extent.width,
					.height = swapchain.extent.height,
					.depth = 1,
				},
				.mipLevels = 1,
				.arrayLayers = 1,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.tiling = VK_IMAGE_TILING_OPTIMAL,
				.usage = (
					VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
					| VK_IMAGE_USAGE_SAMPLED_BIT
				),
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
				.queueFamilyIndexCount = 0,
				.pQueueFamilyIndices = nullptr,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			};
			VmaAllocationCreateInfo const allocCi = {
				.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
				.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
				.requiredFlags = 0,
				.preferredFlags = 0,
				.memoryTypeBits = 0,
				.pool = nullptr,
				.pUserData = nullptr,
				.priority = 0.0f,
			};
			VkAssert(
				vmaCreateImage(
					allocator,
					&imageCi,
					&allocCi,
					&depthImages[i],
					&depthAllocs[i],
					nullptr
				)
			);
			VkImageViewCreateInfo const viewCi = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.pNext = nullptr,
				.flags = 0,
				.image = depthImages[i],
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = VK_FORMAT_D32_SFLOAT,
				.components = VkComponentMapping {
					.r = VK_COMPONENT_SWIZZLE_IDENTITY,
					.g = VK_COMPONENT_SWIZZLE_IDENTITY,
					.b = VK_COMPONENT_SWIZZLE_IDENTITY,
					.a = VK_COMPONENT_SWIZZLE_IDENTITY,
				},
				.subresourceRange = VkImageSubresourceRange {
					.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};
			VkAssert(
				vkCreateImageView(
					device.device, &viewCi, nullptr, &depthImageViews[i]
				)
			);
		}
	}

	// -- command pools
	VkCommandPool commandPoolGraphics;
	{
		VkCommandPoolCreateInfo const ci = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.pNext = nullptr,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = (
				device.get_queue_index(vkb::QueueType::graphics).value()
			),
		};
		VkAssert(
			vkCreateCommandPool(device.device, &ci, nullptr, &commandPoolGraphics)
		);
	}

	VkCommandPool commandPoolCompute;
	{
		VkCommandPoolCreateInfo const ci = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.pNext = nullptr,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = (
				device.get_queue_index(vkb::QueueType::compute).value()
			),
		};
		VkAssert(
			vkCreateCommandPool(device.device, &ci, nullptr, &commandPoolCompute)
		);
	}

	sBindlessSetLayout = create_bindless_set_layout(device.device);

	// [0,128) = root/frame constants; [128,256) = per-draw
	VkPushConstantRange const universalPushConstantRange = {
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256,
	};
	VkPipelineLayoutCreateInfo const universalPipelineLayoutCi = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.setLayoutCount = 1u,
		.pSetLayouts = &sBindlessSetLayout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &universalPushConstantRange,
	};
	VkPipelineLayout pipelineLayoutUniversal;
	VkAssert(
		vkCreatePipelineLayout(
			device.device,
			&universalPipelineLayoutCi,
			nullptr,
			&pipelineLayoutUniversal
		)
	);

	sDevice = new ImplDevice {
		.window = window,
		.instance = instance,
		.physicalDevice = device.physical_device,
		.device = device,
		.swapchain = swapchain,
		.swapchainImages = swapchainImages,
		.swapchainImageViews = swapchainImageViews,
		.swapchainDepthImages = depthImages,
		.swapchainDepthImageViews = depthImageViews,
		.swapchainDepthAllocs = depthAllocs,
		.headlessSwapchainAlloc = headlessColorAlloc,
		.headlessDepthAlloc = headlessDepthAlloc,
		.swapchainImageIndex = 0u,
		.renderExtent = swapchain.extent,
		.commandPoolGraphics = commandPoolGraphics,
		.commandPoolCompute = commandPoolCompute,
		.pipelineLayoutUniversal = pipelineLayoutUniversal,
		.surface = surface,
		.queueGraphics = graphicsQueue,
		.queueCompute = computeQueue,
		.allocator = allocator,
		.frameData = {},
		.frameIndex = 0u,
		.semaphoresRender = {},
		.imagePool = (
			srat::HandlePool<vkof::Image, ImplTexture>::create(1024)
		),
		.bufferPool = (
			srat::HandlePool<vkof::Buffer, ImplBuffer>::create(1024)
		),
		.samplerPool = (
			srat::HandlePool<vkof::Sampler, ImplSampler>::create(1024)
		),
		.transientImagePool = (
			srat::HandlePool<vkof::TransientImage, ImplTransientImage>
				::create(32)
		),
		.transientBufferPool = (
			srat::HandlePool<vkof::TransientBuffer, ImplTransientBuffer>
				::create(32)
		),
		.pipelineGraphicsPool = (
			srat::HandlePool<vkof::Pipeline, ImplPipelineGraphics>
				::create(128)
		),
		.pipelineComputePool = (
			srat::HandlePool<vkof::Pipeline, ImplPipelineCompute>
				::create(128)
		),
		.commandBufferPool = (
			srat::HandlePool<vkof::CommandBuffer, ImplCommandBuffer>
				::create(256)
		),
		.renderNodePool = (
			srat::HandlePool<vkof::RenderNode, ImplRenderNode>
				::create(64)
		),
		.blasPool = (
			srat::HandlePool<
				vkof::AccelerationStructureBlas, ImplAccelerationStructureBlas
			>::create(256)
		),
		.tlasPool = (
			srat::HandlePool<
				vkof::AccelerationStructureTlas, ImplAccelerationStructureTlas
			>::create(16)
		),
		.profiler = {},
	};

	create_bindless_pool_and_set(sDevice->device);
	pfnVkCmdDrawMeshTasksEXT = (
		(PFN_vkCmdDrawMeshTasksEXT)
		vkGetDeviceProcAddr(
			sDevice->device,
			"vkCmdDrawMeshTasksEXT"
		)
	);
	pfnSetDebugName = (
		(PFN_vkSetDebugUtilsObjectNameEXT)
		vkGetDeviceProcAddr(
			sDevice->device,
			"vkSetDebugUtilsObjectNameEXT"
		)
	);

#if defined(VKOF_AFTERMATH)
	sDevice->pfnCmdSetCheckpointNV = (
		(PFN_vkCmdSetCheckpointNV)
		vkGetDeviceProcAddr(
			sDevice->device,
			"vkCmdSetCheckpointNV"
		)
	);
	sDevice->pfnGetQueueCheckpointDataNV = (
		(PFN_vkGetQueueCheckpointDataNV)
		vkGetDeviceProcAddr(
			sDevice->device,
			"vkGetQueueCheckpointDataNV"
		)
	);
	vkof_aftermath_set_handles(
		sDevice->device.device,
		graphicsQueue,
		sDevice->pfnGetQueueCheckpointDataNV
	);
#endif

	// load rt functions
	{
		sDevice->pfnCmdBuildAccelerationStructuresKHR = (
			(PFN_vkCmdBuildAccelerationStructuresKHR)
			vkGetDeviceProcAddr(
				sDevice->device,
				"vkCmdBuildAccelerationStructuresKHR"
			)
		);
		sDevice->pfnGetAccelerationStructureBuildSizesKHR = (
			(PFN_vkGetAccelerationStructureBuildSizesKHR)
			vkGetDeviceProcAddr(
				sDevice->device,
				"vkGetAccelerationStructureBuildSizesKHR"
			)
		);
		sDevice->pfnCreateAccelerationStructureKHR = (
			(PFN_vkCreateAccelerationStructureKHR)
			vkGetDeviceProcAddr(
				sDevice->device,
				"vkCreateAccelerationStructureKHR"
			)
		);
		sDevice->pfnDestroyAccelerationStructureKHR = (
			(PFN_vkDestroyAccelerationStructureKHR)
			vkGetDeviceProcAddr(
				sDevice->device,
				"vkDestroyAccelerationStructureKHR"
			)
		);
		sDevice->pfnGetAccelerationStructureDeviceAddressKHR = (
			(PFN_vkGetAccelerationStructureDeviceAddressKHR)
			vkGetDeviceProcAddr(
				sDevice->device,
				"vkGetAccelerationStructureDeviceAddressKHR"
			)
		);
	}

	// -- per-frame-in-flight: command buffers, semaphores, fences
	{
		VkCommandBufferAllocateInfo const cbai = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.pNext = nullptr,
			.commandPool = sDevice->commandPoolGraphics,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		VkSemaphoreCreateInfo const sci = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
		};
		// pre-signaled so the first frame doesn't wait forever
		VkFenceCreateInfo const fci = {
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.pNext = nullptr,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};
		for (auto & fd : sDevice->frameData) {
			VkAssert(
				vkAllocateCommandBuffers(
					sDevice->device, &cbai, &fd.cmdGraphics
				)
			);
			VkAssert(
				vkCreateSemaphore(
					sDevice->device, &sci, nullptr, &fd.semaphoreAcquire
				)
			);
			VkAssert(
				vkCreateFence(sDevice->device, &fci, nullptr, &fd.fence)
			);
		}
		if (!isHeadless) {
			sDevice->semaphoresRender.resize(sDevice->swapchain.image_count);
			for (VkSemaphore & sem : sDevice->semaphoresRender) {
				VkAssert(
					vkCreateSemaphore(sDevice->device, &sci, nullptr, &sem)
				);
			}
		}
	}
	sDevice->frameIndex = 0u;

	// -- imgui (windowed only)
	if (!isHeadless) {
		ImGui::CreateContext();
		ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		ImGui_ImplGlfw_InitForVulkan(window, true);

		VkDescriptorPoolSize const imguiPoolSize = {
			.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 16,
		};
		VkDescriptorPoolCreateInfo const imguiPoolCi = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.pNext = nullptr,
			.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
			.maxSets = 16,
			.poolSizeCount = 1,
			.pPoolSizes = &imguiPoolSize,
		};
		VkAssert(
			vkCreateDescriptorPool(
				sDevice->device,
				&imguiPoolCi,
				nullptr,
				&sDevice->imguiDescriptorPool
			)
		);

		VkFormat const swapchainFormat = sDevice->swapchain.image_format;
		VkPipelineRenderingCreateInfoKHR const imguiRenderingCi = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
			.pNext = nullptr,
			.viewMask = 0,
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &swapchainFormat,
			.depthAttachmentFormat = VK_FORMAT_UNDEFINED,
			.stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
		};
		ImGui_ImplVulkan_InitInfo imguiVkInfo = {};
		imguiVkInfo.ApiVersion = VK_API_VERSION_1_3;
		imguiVkInfo.Instance = sDevice->instance.instance;
		imguiVkInfo.PhysicalDevice = sDevice->device.physical_device;
		imguiVkInfo.Device = sDevice->device.device;
		imguiVkInfo.QueueFamily = (
			sDevice->device.get_queue_index(vkb::QueueType::graphics).value()
		);
		imguiVkInfo.Queue = sDevice->queueGraphics;
		imguiVkInfo.DescriptorPool = sDevice->imguiDescriptorPool;
		imguiVkInfo.MinImageCount = 2;
		imguiVkInfo.ImageCount = (u32)sDevice->swapchainImages.size();
		imguiVkInfo.UseDynamicRendering = true;
		imguiVkInfo.PipelineInfoMain.PipelineRenderingCreateInfo = imguiRenderingCi;
		ImGui_ImplVulkan_Init(&imguiVkInfo);
	}

	if (!isHeadless) {
		std::filesystem::path const vkofDir = (
			std::filesystem::canonical(
				std::filesystem::path(__FILE__).parent_path()
			)
		);
		std::string const meshPath = (vkofDir / "debug_line.mesh").string();
		std::string const fragPath = (vkofDir / "debug_line.frag").string();
		sDebugLinePipelineWithDepth = (
			build_debug_line_pipeline(meshPath, fragPath, true)
		);
		sDebugLinePipelineNoDepth = (
			build_debug_line_pipeline(meshPath, fragPath, false)
		);
		sDebugVertexBuffer = vkof::buffer_create({
			.byteCount = sizeof(DebugVertex) * skDebugMaxLines * 2u,
			.memory = vkof::BufferMemory::HostWritable,
		});
		sDebugSphereCapacity = skDebugSphereInitialCapacity;
		sDebugSphereBuffer = vkof::buffer_create({
			.byteCount = sizeof(vkof::DebugSphere) * sDebugSphereCapacity,
			.memory = vkof::BufferMemory::HostWritable,
		});
	}

	profiler_init();
}

void vkof::init()
{
	init_impl(false, 1920u, 1080u);
}

void vkof::init_headless(u32 const width, u32 const height)
{
	init_impl(true, width, height);
}

void vkof::device_wait_idle()
{
	vkDeviceWaitIdle(sDevice->device);
}

void vkof::shutdown()
{
	vkDeviceWaitIdle(sDevice->device);

	if (sDevice->profiler.queryPool != VK_NULL_HANDLE) {
		vkDestroyQueryPool(sDevice->device, sDevice->profiler.queryPool, nullptr);
	}

	if (sDebugLinePipelineWithDepth != VK_NULL_HANDLE) {
		vkDestroyPipeline(
			sDevice->device, sDebugLinePipelineWithDepth, nullptr
		);
		sDebugLinePipelineWithDepth = VK_NULL_HANDLE;
	}
	if (sDebugLinePipelineNoDepth != VK_NULL_HANDLE) {
		vkDestroyPipeline(
			sDevice->device, sDebugLinePipelineNoDepth, nullptr
		);
		sDebugLinePipelineNoDepth = VK_NULL_HANDLE;
	}
	if (sDebugVertexBuffer.id != 0u) {
		vkof::buffer_destroy(sDebugVertexBuffer);
		sDebugVertexBuffer = { 0 };
	}
	sDebugVertices.clear();
	if (sDebugSphereBuffer.id != 0u) {
		vkof::buffer_destroy(sDebugSphereBuffer);
		sDebugSphereBuffer = { 0 };
	}
	sDebugSphereData.clear();
	sDebugSphereBatches.clear();

	if (sDevice->window != nullptr) {
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		vkDestroyDescriptorPool(
			sDevice->device, sDevice->imguiDescriptorPool, nullptr
		);
	}

	// -- destroy transient resources (owned by the transient pools)
	for (auto const & ti : sDevice->transientImageHandles) {
		auto const impl = sDevice->transientImagePool.get(ti);
		if (!impl) { continue; }
		for (u32 slot = 0u; slot < 2u; ++slot) {
			if (impl->handles[slot].id != 0u) {
				vkof::image_destroy(impl->handles[slot]);
			}
		}
	}
	for (auto const & h : sDevice->transientBufferCleanupHandles) {
		vkof::buffer_destroy(h);
	}

	// -- destroy per-frame and per-image sync primitives
	// (command buffers are freed when commandPoolGraphics is destroyed)
	for (auto & fd : sDevice->frameData) {
		vkDestroySemaphore(sDevice->device, fd.semaphoreAcquire, nullptr);
		vkDestroyFence(sDevice->device, fd.fence, nullptr);
	}
	for (VkSemaphore sem : sDevice->semaphoresRender) {
		vkDestroySemaphore(sDevice->device, sem, nullptr);
	}

	for (VkImageView view : sDevice->swapchainDepthImageViews) {
		vkDestroyImageView(sDevice->device, view, nullptr);
	}
	if (sDevice->window == nullptr) {
		// headless: depth and color images are VMA-allocated; use stored allocs
		vmaDestroyImage(
			sDevice->allocator,
			sDevice->swapchainDepthImages[0],
			sDevice->headlessDepthAlloc
		);
		for (VkImageView view : sDevice->swapchainImageViews) {
			vkDestroyImageView(sDevice->device, view, nullptr);
		}
		vmaDestroyImage(
			sDevice->allocator,
			sDevice->swapchainImages[0],
			sDevice->headlessSwapchainAlloc
		);
	} else {
		for (size_t i = 0; i < sDevice->swapchainDepthImages.size(); ++i) {
			vmaDestroyImage(
				sDevice->allocator,
				sDevice->swapchainDepthImages[i],
				sDevice->swapchainDepthAllocs[i]
			);
		}
	}

	for (const auto & id : sDevice->pipelineGraphicsHandles) {
		auto const & pipeline = sDevice->pipelineGraphicsPool.get(id);
		vkDestroyPipeline(sDevice->device, pipeline->pipeline, nullptr);
	}
	for (auto const & id : sDevice->pipelineComputeHandles) {
		auto const & pipeline = sDevice->pipelineComputePool.get(id);
		vkDestroyPipeline(sDevice->device, pipeline->pipeline, nullptr);
	}

	vmaDestroyAllocator(sDevice->allocator);
	vkDestroyDescriptorPool(sDevice->device, sBindlessPool, nullptr);
	vkDestroyDescriptorSetLayout(sDevice->device, sBindlessSetLayout, nullptr);
	sBindlessPool = VK_NULL_HANDLE;
	sBindlessSet = VK_NULL_HANDLE;
	sBindlessSetLayout = VK_NULL_HANDLE;
	sBindlessSamplerNextSlot = 1u;
	sBindlessStorageUintNextSlot = 1u;
	sBindlessStorageFloatNextSlot = 1u;
	sTransientStorageSlotCache.clear();
	vkDestroyPipelineLayout(
		sDevice->device, sDevice->pipelineLayoutUniversal, nullptr
	);
	vkDestroyCommandPool(sDevice->device, sDevice->commandPoolGraphics, nullptr);
	vkDestroyCommandPool(sDevice->device, sDevice->commandPoolCompute, nullptr);

	if (sDevice->window != nullptr) {
		// windowed: let vkb destroy swapchain image views, swapchain, surface
		sDevice->swapchain.destroy_image_views(sDevice->swapchainImageViews);
		vkb::destroy_swapchain(sDevice->swapchain);
		vkb::destroy_surface(sDevice->instance, sDevice->surface);
	}
#if defined(VKOF_AFTERMATH)
	vkof_aftermath_disable();
#endif
	vkb::destroy_device(sDevice->device);
	vkb::destroy_instance(sDevice->instance);
	if (sDevice->window != nullptr) {
		glfwDestroyWindow(sDevice->window);
		glfwTerminate();
	}

	delete sDevice;
	sDevice = nullptr;
}

// -----------------------------------------------------------------------------
// -- pipeline creation
// -----------------------------------------------------------------------------

static VkPipeline build_compute_pipeline(
	ImplPipelineCompute const & impl
)
{
	VkShaderModule const computeShaderModule = (
		compile_and_load_shader_module(
			impl.pathCompute, ShaderStage::compute,
			impl.defines, impl.includePaths
		)
	);
	if (computeShaderModule == VK_NULL_HANDLE) {
		return VK_NULL_HANDLE;
	}
	VkPipelineShaderStageCreateInfo const shaderStageCi = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.stage = VK_SHADER_STAGE_COMPUTE_BIT,
		.module = computeShaderModule,
		.pName = "main",
		.pSpecializationInfo = nullptr,
	};
	VkComputePipelineCreateInfo const pipelineCi = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.stage = shaderStageCi,
		.layout = sDevice->pipelineLayoutUniversal,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1,
	};
	VkPipeline pipeline;
	VkResult const result = vkCreateComputePipelines(
		sDevice->device,
		VK_NULL_HANDLE, // pipeline cache
		1, // createInfoCount
		&pipelineCi,
		nullptr, // allocator
		&pipeline
	);
	vkDestroyShaderModule(sDevice->device, computeShaderModule, nullptr);
	if (result != VK_SUCCESS) {
		printf("failed to create compute pipeline: %d\n", result);
		return VK_NULL_HANDLE;
	}
	return pipeline;
}

static VkPipeline build_graphics_pipeline(
	ImplPipelineGraphics const & impl
)
{
	// -- compile mesh + fragment from GLSL source
	VkShaderModule const meshModule = (
		compile_and_load_shader_module(
			impl.pathMesh, ShaderStage::mesh,
			impl.defines, impl.includePaths
		)
	);
	VkShaderModule const fragmentModule = (
		compile_and_load_shader_module(
			impl.pathFragment, ShaderStage::fragment,
			impl.defines, impl.includePaths
		)
	);
	if (meshModule == VK_NULL_HANDLE || fragmentModule == VK_NULL_HANDLE) {
		// keep-old-on-failure: bail before creating anything.
		if (meshModule != VK_NULL_HANDLE) {
			vkDestroyShaderModule(sDevice->device, meshModule, nullptr);
		}
		if (fragmentModule != VK_NULL_HANDLE) {
			vkDestroyShaderModule(sDevice->device, fragmentModule, nullptr);
		}
		return VK_NULL_HANDLE;
	}

	// -- dynamic rendering attachment formats
	std::vector<VkFormat> colorFormats;
	colorFormats.reserve(impl.colorAttachmentFormats.size());
	for (vkof::ImageFormat f : impl.colorAttachmentFormats) {
		colorFormats.push_back(to_vk_format(f));
	}
	VkFormat const depthFormat = to_vk_format(impl.depthAttachmentFormat);

	VkPipelineRenderingCreateInfo const renderingCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.pNext = nullptr,
		.viewMask = 0,
		.colorAttachmentCount = (u32)colorFormats.size(),
		.pColorAttachmentFormats = colorFormats.data(),
		.depthAttachmentFormat = depthFormat,
		.stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
	};

	// -- mesh + fragment stages (vertex-pulling model: no vertex stage)
	VkPipelineShaderStageCreateInfo const stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.stage = VK_SHADER_STAGE_MESH_BIT_EXT,
			.module = meshModule,
			.pName = "main",
			.pSpecializationInfo = nullptr,
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragmentModule,
			.pName = "main",
			.pSpecializationInfo = nullptr,
		},
	};

	VkPipelineViewportStateCreateInfo const viewportState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = nullptr,
		.scissorCount = 1,
		.pScissors = nullptr,
	};

	VkPipelineRasterizationStateCreateInfo const rasterState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = to_vk_cull(impl.cullMode),
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.depthBiasConstantFactor = 0.0f,
		.depthBiasClamp = 0.0f,
		.depthBiasSlopeFactor = 0.0f,
		.lineWidth = 1.0f,
	};

	VkPipelineMultisampleStateCreateInfo const multisampleState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.sampleShadingEnable = VK_FALSE,
		.minSampleShading = 1.0f,
		.pSampleMask = nullptr,
		.alphaToCoverageEnable = VK_FALSE,
		.alphaToOneEnable = VK_FALSE,
	};

	bool const alpha = (impl.blendMode == vkof::BlendMode::alpha_blend);
	VkPipelineColorBlendAttachmentState const blendAttachment = {
		.blendEnable = (alpha ? VK_TRUE : VK_FALSE),
		.srcColorBlendFactor = (
			alpha ? VK_BLEND_FACTOR_SRC_ALPHA : VK_BLEND_FACTOR_ONE
		),
		.dstColorBlendFactor = (
			alpha
				? VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA
				: VK_BLEND_FACTOR_ZERO
		),
		.colorBlendOp = VK_BLEND_OP_ADD,
		.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
		.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
		.alphaBlendOp = VK_BLEND_OP_ADD,
		.colorWriteMask = (
			  VK_COLOR_COMPONENT_R_BIT
			| VK_COLOR_COMPONENT_G_BIT
			| VK_COLOR_COMPONENT_B_BIT
			| VK_COLOR_COMPONENT_A_BIT
		),
	};
	std::vector<VkPipelineColorBlendAttachmentState> blendAttachments(
		colorFormats.size(), blendAttachment
	);
	VkPipelineColorBlendStateCreateInfo const blendState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_COPY,
		.attachmentCount = (u32)blendAttachments.size(),
		.pAttachments = blendAttachments.data(),
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f },
	};

	VkBool32 depthTest { VK_FALSE };
	VkBool32 depthWrite { VK_FALSE };
	depth_test_flags(impl.depthTest, depthTest, depthWrite);
	VkPipelineDepthStencilStateCreateInfo const depthState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.depthTestEnable = depthTest,
		.depthWriteEnable = depthWrite,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
		.front = {},
		.back = {},
		.minDepthBounds = 0.0f,
		.maxDepthBounds = 1.0f,
	};

	VkDynamicState const dynamicStates[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	VkPipelineDynamicStateCreateInfo const dynamicState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamicStates,
	};

	VkGraphicsPipelineCreateInfo const createInfo = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = &renderingCreateInfo,
		.flags = 0,
		.stageCount = 2,
		.pStages = stages,
		// mesh pipeline: no vertex input / input assembly
		.pVertexInputState = nullptr,
		.pInputAssemblyState = nullptr,
		.pTessellationState = nullptr,
		.pViewportState = &viewportState,
		.pRasterizationState = &rasterState,
		.pMultisampleState = &multisampleState,
		.pDepthStencilState = &depthState,
		.pColorBlendState = &blendState,
		.pDynamicState = &dynamicState,
		.layout = sDevice->pipelineLayoutUniversal,
		.renderPass = VK_NULL_HANDLE,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1,
	};

	VkPipeline pipeline;
	VkAssert(
		vkCreateGraphicsPipelines(
			sDevice->device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &pipeline
		)
	);

	// modules consumed by pipeline creation; safe to destroy now
	vkDestroyShaderModule(sDevice->device, meshModule, nullptr);
	vkDestroyShaderModule(sDevice->device, fragmentModule, nullptr);
	return pipeline;
}

static VkPipeline build_debug_line_pipeline(
	std::string const & meshPath,
	std::string const & fragPath,
	bool const withDepth
) {
	std::vector<std::string> const emptyVec;
	VkShaderModule const meshModule = compile_and_load_shader_module(
		meshPath, ShaderStage::mesh, emptyVec, emptyVec
	);
	VkShaderModule const fragModule = compile_and_load_shader_module(
		fragPath, ShaderStage::fragment, emptyVec, emptyVec
	);
	if (meshModule == VK_NULL_HANDLE || fragModule == VK_NULL_HANDLE) {
		if (meshModule != VK_NULL_HANDLE) {
			vkDestroyShaderModule(sDevice->device, meshModule, nullptr);
		}
		if (fragModule != VK_NULL_HANDLE) {
			vkDestroyShaderModule(sDevice->device, fragModule, nullptr);
		}
		return VK_NULL_HANDLE;
	}

	VkFormat const colorFmt = sDevice->swapchain.image_format;
	VkFormat const depthFmt = (
		withDepth ? VK_FORMAT_D24_UNORM_S8_UINT : VK_FORMAT_UNDEFINED
	);

	VkPipelineRenderingCreateInfo const renderingCi = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.pNext = nullptr,
		.viewMask = 0,
		.colorAttachmentCount = 1,
		.pColorAttachmentFormats = &colorFmt,
		.depthAttachmentFormat = depthFmt,
		.stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
	};

	VkPipelineShaderStageCreateInfo const stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.stage = VK_SHADER_STAGE_MESH_BIT_EXT,
			.module = meshModule,
			.pName = "main",
			.pSpecializationInfo = nullptr,
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragModule,
			.pName = "main",
			.pSpecializationInfo = nullptr,
		},
	};

	VkPipelineViewportStateCreateInfo const viewportState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = nullptr,
		.scissorCount = 1,
		.pScissors = nullptr,
	};

	VkPipelineRasterizationStateCreateInfo const rasterState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.depthBiasConstantFactor = 0.0f,
		.depthBiasClamp = 0.0f,
		.depthBiasSlopeFactor = 0.0f,
		.lineWidth = 1.0f,
	};

	VkPipelineMultisampleStateCreateInfo const multisampleState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.sampleShadingEnable = VK_FALSE,
		.minSampleShading = 1.0f,
		.pSampleMask = nullptr,
		.alphaToCoverageEnable = VK_FALSE,
		.alphaToOneEnable = VK_FALSE,
	};

	VkPipelineColorBlendAttachmentState const blendAtt = {
		.blendEnable = VK_FALSE,
		.srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
		.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
		.colorBlendOp = VK_BLEND_OP_ADD,
		.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
		.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
		.alphaBlendOp = VK_BLEND_OP_ADD,
		.colorWriteMask = (
			VK_COLOR_COMPONENT_R_BIT
			| VK_COLOR_COMPONENT_G_BIT
			| VK_COLOR_COMPONENT_B_BIT
			| VK_COLOR_COMPONENT_A_BIT
		),
	};

	VkPipelineColorBlendStateCreateInfo const blendState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_COPY,
		.attachmentCount = 1,
		.pAttachments = &blendAtt,
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f },
	};

	VkPipelineDepthStencilStateCreateInfo const depthState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.depthTestEnable = withDepth ? VK_TRUE : VK_FALSE,
		.depthWriteEnable = VK_FALSE,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
		.front = {},
		.back = {},
		.minDepthBounds = 0.0f,
		.maxDepthBounds = 1.0f,
	};

	VkDynamicState const dynamicStates[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	VkPipelineDynamicStateCreateInfo const dynamicState = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamicStates,
	};

	VkGraphicsPipelineCreateInfo const pipelineCi = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = &renderingCi,
		.flags = 0,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = nullptr,
		.pInputAssemblyState = nullptr,
		.pTessellationState = nullptr,
		.pViewportState = &viewportState,
		.pRasterizationState = &rasterState,
		.pMultisampleState = &multisampleState,
		.pDepthStencilState = &depthState,
		.pColorBlendState = &blendState,
		.pDynamicState = &dynamicState,
		.layout = sDevice->pipelineLayoutUniversal,
		.renderPass = VK_NULL_HANDLE,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1,
	};

	VkPipeline pipeline;
	VkResult const result = vkCreateGraphicsPipelines(
		sDevice->device, VK_NULL_HANDLE, 1, &pipelineCi, nullptr, &pipeline
	);
	vkDestroyShaderModule(sDevice->device, meshModule, nullptr);
	vkDestroyShaderModule(sDevice->device, fragModule, nullptr);
	if (result != VK_SUCCESS) {
		printf("[vkof] failed to create debug line pipeline: %d\n", result);
		return VK_NULL_HANDLE;
	}
	return pipeline;
}

void vkof::debug_draw_spheres(
	srat::slice<vkof::DebugSphere const> const & spheres,
	vkof::Pipeline const pipeline
) {
	u32 const startIdx = (u32)sDebugSphereData.size();
	u32 const count = (u32)spheres.size();
	if (count == 0u) { return; }
	for (u32 i = 0u; i < count; ++i) {
		sDebugSphereData.push_back(spheres.ptr()[i]);
	}
	sDebugSphereBatches.push_back({
		.startIdx = startIdx,
		.count = count,
		.pipeline = pipeline,
	});
}

vkof::Pipeline vkof::debug_sphere_pipeline_create(
	vkof::DebugSpherePipelineCreateInfo const & createInfo
) {
	std::filesystem::path const vkofDir = (
		std::filesystem::canonical(
			std::filesystem::path(__FILE__).parent_path()
		)
	);
	std::string const meshPath = (
		(vkofDir / "debug_sphere.mesh").string()
	);
	static constexpr vkof::ImageFormat kColorFmt = (
		vkof::ImageFormat::b8g8r8a8_unorm
	);
	return vkof::pipeline_graphics_create({
		.pathMesh = meshPath.c_str(),
		.pathFragment = createInfo.pathFrag,
		.attachmentColorFormats = srat::slice<vkof::ImageFormat const>(
			&kColorFmt, 1u
		),
		.attachmentDepthStencilFormat = (
			vkof::ImageFormat::d24_unorm_s8_uint
		),
		.depthTest = vkof::DepthTest::write_on_test_on,
		.cullMode = vkof::CullMode::none,
		.blendMode = vkof::BlendMode::none,
		.defines = createInfo.defines,
		.includePaths = createInfo.includePaths,
	});
}

// -----------------------------------------------------------------------------
// -- shader compilation
// -----------------------------------------------------------------------------

// last compile error, surfaced so a failed hot reload can be shown
// rather than crashing the session.
static std::string sLastCompileError;

static char const * shader_stage_flag(ShaderStage const & stage) {
	switch (stage) {
		case ShaderStage::fragment: return "frag";
		case ShaderStage::compute:  return "comp";
		case ShaderStage::mesh:     return "mesh";
	}
	return "vert";
}

static CompileResult system_capture(char const * cmd) {
	std::string output;
	FILE * pipe = popen(cmd, "r");
	if (!pipe) { return { -1, "failed to launch compiler" }; }
	char buffer[256];
	while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
		output += buffer;
	}
	int const status = pclose(pipe);
	return { status, output };
}

// compiles glsl -> spv, returns the module. on failure returns
// VK_NULL_HANDLE and leaves sLastCompileError set (caller decides whether
// to keep the old pipeline or abort).
static VkShaderModule compile_and_load_shader_module(
	std::string const & glslPath,
	ShaderStage stage,
	std::vector<std::string> const & defines,
	std::vector<std::string> const & includePaths
) {
	std::string const shaderDir = (
		glslPath.substr(0, glslPath.rfind('/') + 1)
	);
	std::string const baseName = (
		glslPath.substr(glslPath.rfind('/') + 1)
	);
	std::string const binDir = shaderDir + "bin";
	std::string const spvPath = binDir + "/" + baseName + ".spv";
	std::filesystem::create_directories(binDir);

	std::string cmd = (
		std::string("glslangValidator -S ")
		+ shader_stage_flag(stage)
		+ " --target-env vulkan1.3"
#if defined(VKOF_AFTERMATH)
		+ " -gVS"
#endif
	);
	for (std::string const & d : defines) {
		cmd += " -D" + d;
	}
	for (std::string const & p : includePaths) {
		cmd += " -I" + p;
	}
	cmd += " -o " + spvPath + " " + glslPath;
	CompileResult const result = system_capture(cmd.c_str());
	if (result.exitCode != 0) {
		printf(
			"failed to compile shader %s:\n%s\n",
			glslPath.c_str(), result.output.c_str()
		);
		sLastCompileError = result.output;
		return VK_NULL_HANDLE;
	}
	sLastCompileError.clear();

	FILE * f = fopen(spvPath.c_str(), "rb");
	if (!f) {
		printf("failed to open compiled spv: %s\n", spvPath.c_str());
		return VK_NULL_HANDLE;
	}
	fseek(f, 0, SEEK_END);
	size_t const size = ftell(f);
	fseek(f, 0, SEEK_SET);
	std::vector<u8> code(size);
	fread(code.data(), 1, size, f);
	fclose(f);

#if defined(VKOF_AFTERMATH)
	vkof_aftermath_register_spirv(code.data(), (uint32_t)code.size());
#endif

	VkShaderModuleCreateInfo const ci = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.codeSize = code.size(),
		.pCode = (u32 const *)code.data(),
	};
	VkShaderModule mod;
	VkAssert(vkCreateShaderModule(sDevice->device, &ci, nullptr, &mod));
	return mod;
}

// -----------------------------------------------------------------------------
// -- shader dependency tracking
// -----------------------------------------------------------------------------

static void collect_deps_impl(
	std::string const & path,
	std::vector<std::string> const & includePaths,
	std::unordered_set<std::string> & visited,
	std::vector<std::string> & result
) {
	if (path.empty()) { return; }
	std::ifstream file(path);
	if (!file.is_open()) { return; }

	std::filesystem::path const fileDir = (
		std::filesystem::path(path).parent_path()
	);
	std::string line;
	while (std::getline(file, line)) {
		size_t start = 0u;
		while (start < line.size() && (line[start] == ' ' || line[start] == '\t')) {
			++start;
		}
		if (start >= line.size() || line[start] != '#') { continue; }
		++start;
		while (start < line.size() && (line[start] == ' ' || line[start] == '\t')) {
			++start;
		}
		if (line.compare(start, 7u, "include") != 0) { continue; }
		start += 7u;
		while (start < line.size() && (line[start] == ' ' || line[start] == '\t')) {
			++start;
		}
		if (start >= line.size()) { continue; }

		char const openDelim = line[start];
		bool const isQuoted = (openDelim == '"');
		if (openDelim != '"' && openDelim != '<') { continue; }
		char const closeDelim = isQuoted ? '"' : '>';
		++start;
		size_t const end = line.find(closeDelim, start);
		if (end == std::string::npos) { continue; }

		std::string const includeName = line.substr(start, end - start);
		if (includeName.empty()) { continue; }

		std::string resolvedPath;
		if (isQuoted) {
			std::error_code ec;
			std::filesystem::path const canon = std::filesystem::canonical(
				fileDir / includeName, ec
			);
			if (!ec) { resolvedPath = canon.string(); }
		}
		if (resolvedPath.empty()) {
			for (std::string const & incPath : includePaths) {
				std::error_code ec;
				std::filesystem::path const canon = std::filesystem::canonical(
					std::filesystem::path(incPath) / includeName, ec
				);
				if (!ec) {
					resolvedPath = canon.string();
					break;
				}
			}
		}
		if (resolvedPath.empty()) { continue; }
		if (visited.count(resolvedPath)) { continue; }
		visited.insert(resolvedPath);
		result.push_back(resolvedPath);
		collect_deps_impl(resolvedPath, includePaths, visited, result);
	}
}

static std::vector<std::string> collect_shader_dependencies(
	std::vector<std::string> const & primaryPaths,
	std::vector<std::string> const & includePaths
) {
	std::unordered_set<std::string> visited;
	for (std::string const & p : primaryPaths) {
		if (!p.empty()) { visited.insert(p); }
	}
	std::vector<std::string> result;
	for (std::string const & p : primaryPaths) {
		collect_deps_impl(p, includePaths, visited, result);
	}
	return result;
}

static std::vector<std::filesystem::file_time_type> snapshot_write_times(
	std::vector<std::string> const & paths
) {
	std::vector<std::filesystem::file_time_type> times;
	times.reserve(paths.size());
	for (std::string const & p : paths) {
		times.push_back(file_write_time(p));
	}
	return times;
}

static bool any_dependency_changed(
	std::vector<std::string> const & paths,
	std::vector<std::filesystem::file_time_type> const & times
) {
	for (size_t i = 0u; i < paths.size(); ++i) {
		if (file_changed(paths[i], times[i])) { return true; }
	}
	return false;
}

// -----------------------------------------------------------------------------
// -- shader hot reload
// -----------------------------------------------------------------------------

static bool file_changed(
	std::string const & path,
	std::filesystem::file_time_type const & lastWriteTime
) {
	return file_write_time(path) != lastWriteTime;
}

static void pipeline_hot_reload()
{
	// -- check if compute or graphics pipeilne changed
	bool anyChanged = false;
	for (auto const & id : sDevice->pipelineGraphicsHandles) {
		ImplPipelineGraphics const & impl = (
			*sDevice->pipelineGraphicsPool.get(id)
		);
		if (file_changed(impl.pathFragment, impl.lastWriteTimeFragment)) {
			anyChanged = true;
			break;
		}
		if (file_changed(impl.pathMesh, impl.lastWriteTimeMesh)) {
			anyChanged = true;
			break;
		}
		if (any_dependency_changed(impl.dependencyPaths, impl.dependencyWriteTimes)) {
			anyChanged = true;
			break;
		}
	}
	if (!anyChanged) {
		for (auto const & id : sDevice->pipelineComputeHandles) {
			ImplPipelineCompute const & impl = (
				*sDevice->pipelineComputePool.get(id)
			);
			if (file_changed(impl.pathCompute, impl.lastWriteTimeCompute)) {
				anyChanged = true;
				break;
			}
			if (
				any_dependency_changed(
					impl.dependencyPaths,
					impl.dependencyWriteTimes
				)
			) {
				anyChanged = true;
				break;
			}
		}
	}

	if (!anyChanged) {
		return;
	}

	// -- reload everything
	vkDeviceWaitIdle(sDevice->device);

	// for (auto & [id, impl] : sDevice->pipelineGraphicsPool) {
	for (auto const & id : sDevice->pipelineGraphicsHandles) {
		ImplPipelineGraphics & impl = (
			*sDevice->pipelineGraphicsPool.get(id)
		);
		bool const vtxChanged = (
			file_changed(impl.pathMesh, impl.lastWriteTimeMesh)
		);
		bool const frgChanged = (
			file_changed(impl.pathFragment, impl.lastWriteTimeFragment)
		);
		bool const depChanged = any_dependency_changed(
			impl.dependencyPaths, impl.dependencyWriteTimes
		);
		if (!vtxChanged && !frgChanged && !depChanged) { continue; }

		VkPipeline const rebuilt = build_graphics_pipeline(impl);
		if (rebuilt == VK_NULL_HANDLE) {
			// compile failed: keep the old pipeline, do NOT advance the
			// tracked write times, so the next save retries automatically.
			printf(
				"hot reload kept old pipeline '%s/%s' (compile failed)\n",
				impl.pathMesh.c_str(), impl.pathFragment.c_str()
			);
			continue;
		}

		VkPipeline const old = impl.pipeline;
		impl.pipeline = rebuilt;
		impl.lastWriteTimeMesh = file_write_time(impl.pathMesh);
		impl.lastWriteTimeFragment = file_write_time(impl.pathFragment);
		impl.dependencyPaths = collect_shader_dependencies(
			{impl.pathMesh, impl.pathFragment}, impl.includePaths
		);
		impl.dependencyWriteTimes = snapshot_write_times(impl.dependencyPaths);
		vkDestroyPipeline(sDevice->device, old, nullptr);
		sShaderReloaded = true;
		printf(
			"hot reloaded pipeline '%s/%s'\n",
			impl.pathMesh.c_str(), impl.pathFragment.c_str()
		);
	}

	// -- compute
	for (auto const & id : sDevice->pipelineComputeHandles) {
		ImplPipelineCompute & impl = (
			*sDevice->pipelineComputePool.get(id)
		);
		bool const primaryChanged = file_changed(
			impl.pathCompute, impl.lastWriteTimeCompute
		);
		bool const depChangedCompute = any_dependency_changed(
			impl.dependencyPaths, impl.dependencyWriteTimes
		);
		if (!primaryChanged && !depChangedCompute) { continue; }

		VkPipeline const rebuilt = build_compute_pipeline(impl);
		if (rebuilt == VK_NULL_HANDLE) {
			printf(
				"hot reload kept old compute pipeline '%s' (compile failed)\n",
				impl.pathCompute.c_str()
			);
			continue;
		}

		VkPipeline const old = impl.pipeline;
		impl.pipeline = rebuilt;
		impl.lastWriteTimeCompute = file_write_time(impl.pathCompute);
		impl.dependencyPaths = collect_shader_dependencies(
			{impl.pathCompute}, impl.includePaths
		);
		impl.dependencyWriteTimes = snapshot_write_times(impl.dependencyPaths);
		vkDestroyPipeline(sDevice->device, old, nullptr);
		sShaderReloaded = true;
		printf("hot reloaded compute pipeline '%s'\n", impl.pathCompute.c_str());
	}
}

// -----------------------------------------------------------------------------
// -- pipeline creation
// -----------------------------------------------------------------------------

vkof::Pipeline vkof::pipeline_graphics_create(
	PipelineGraphicsCreateInfo const & createInfo
) {
	std::vector<vkof::ImageFormat> const colorFormats(
		/*__first=*/ createInfo.attachmentColorFormats.ptr(),
		/*__last=*/ (
			  createInfo.attachmentColorFormats.ptr()
			+ createInfo.attachmentColorFormats.size()
		)
	);
	auto implPipeline = ImplPipelineGraphics {
		.pipeline = VK_NULL_HANDLE,
		.colorAttachmentFormats = std::move(colorFormats),
		.depthAttachmentFormat = createInfo.attachmentDepthStencilFormat,
		.depthTest = createInfo.depthTest,
		.cullMode = createInfo.cullMode,
		.blendMode = createInfo.blendMode,

		.pathFragment = createInfo.pathFragment,
		.pathMesh = createInfo.pathMesh,
		.defines = std::vector<std::string>(
			createInfo.defines.ptr(),
			createInfo.defines.ptr() + createInfo.defines.size()
		),
		.includePaths = std::vector<std::string>(
			createInfo.includePaths.ptr(),
			createInfo.includePaths.ptr() + createInfo.includePaths.size()
		),

		.lastWriteTimeFragment = file_write_time(createInfo.pathFragment),
		.lastWriteTimeMesh = file_write_time(createInfo.pathMesh),

		.dependencyPaths = {},
		.dependencyWriteTimes = {},
	};
	implPipeline.pipeline = build_graphics_pipeline(implPipeline);
	if (implPipeline.pipeline == VK_NULL_HANDLE) {
		std::error_code ec;
		bool const meshOk = (
			implPipeline.pathMesh.empty()
			|| (std::filesystem::exists(implPipeline.pathMesh, ec) && !ec)
		);
		bool const fragOk = (
			implPipeline.pathFragment.empty()
			|| (std::filesystem::exists(implPipeline.pathFragment, ec) && !ec)
		);
		if (!meshOk || !fragOk) {
			printf(
				"failed to create graphics pipeline '%s | %s'\n",
				createInfo.pathMesh, createInfo.pathFragment
			);
			return vkof::Pipeline { .id = 0 };
		}
		printf(
			"graphics pipeline compile failed '%s | %s'; will retry on next save\n",
			createInfo.pathMesh, createInfo.pathFragment
		);
	}
	implPipeline.dependencyPaths = collect_shader_dependencies(
		{implPipeline.pathMesh, implPipeline.pathFragment},
		implPipeline.includePaths
	);
	implPipeline.dependencyWriteTimes = snapshot_write_times(
		implPipeline.dependencyPaths
	);
	vkof::Pipeline const handle = (
		sDevice->pipelineGraphicsPool.allocate(implPipeline)
	);
	sDevice->pipelineGraphicsHandles.emplace_back(handle.id);
	return handle;
}

vkof::Pipeline vkof::pipeline_compute_create(
	vkof::PipelineComputeCreateInfo const & createInfo
) {
	ImplPipelineCompute implPipeline;
	implPipeline.pathCompute = createInfo.pathCompute;
	implPipeline.defines = std::vector<std::string>(
		createInfo.defines.ptr(),
		createInfo.defines.ptr() + createInfo.defines.size()
	);
	implPipeline.includePaths = std::vector<std::string>(
		createInfo.includePaths.ptr(),
		createInfo.includePaths.ptr() + createInfo.includePaths.size()
	);
	implPipeline.layout = sDevice->pipelineLayoutUniversal;
	implPipeline.pipeline = build_compute_pipeline(implPipeline);
	if (implPipeline.pipeline == VK_NULL_HANDLE) {
		std::error_code ec;
		if (!std::filesystem::exists(implPipeline.pathCompute, ec) || ec) {
			printf(
				"failed to create compute pipeline '%s'\n",
				createInfo.pathCompute
			);
			return vkof::Pipeline { .id = 0 };
		}
		printf(
			"compute pipeline compile failed '%s'; will retry on next save\n",
			createInfo.pathCompute
		);
	}
	implPipeline.lastWriteTimeCompute = (
		file_write_time(implPipeline.pathCompute)
	);
	implPipeline.dependencyPaths = collect_shader_dependencies(
		{implPipeline.pathCompute}, implPipeline.includePaths
	);
	implPipeline.dependencyWriteTimes = snapshot_write_times(
		implPipeline.dependencyPaths
	);
	auto const handle = sDevice->pipelineComputePool.allocate(implPipeline);
	sDevice->pipelineComputeHandles.emplace_back(handle.id);
	return handle;
}

void vkof::pipeline_destroy(Pipeline const & pipeline)
{
	// try graphics pool first
	if (auto const impl = sDevice->pipelineGraphicsPool.get(pipeline)) {
		vkDestroyPipeline(sDevice->device, impl->pipeline, nullptr);
		sDevice->pipelineGraphicsPool.free(pipeline);
		auto & v = sDevice->pipelineGraphicsHandles;
		v.erase(
			std::remove_if(
				v.begin(), v.end(),
				[&](vkof::Pipeline const & h) {
					return h.id == pipeline.id;
				}
			),
			v.end()
		);
		return;
	}
	// then compute pool
	if (auto const impl = sDevice->pipelineComputePool.get(pipeline)) {
		vkDestroyPipeline(sDevice->device, impl->pipeline, nullptr);
		sDevice->pipelineComputePool.free(pipeline);
		auto & v = sDevice->pipelineComputeHandles;
		v.erase(
			std::remove_if(
				v.begin(), v.end(),
				[&](vkof::Pipeline const & h) {
					return h.id == pipeline.id;
				}
			),
			v.end()
		);
		return;
	}
}

// -----------------------------------------------------------------------------
// -- draw/dispatch/copy
// -----------------------------------------------------------------------------

void vkof::cmd_draw(CmdDraw const & draw)
{
	auto const implCmd = sDevice->commandBufferPool.get(draw.cmd);
	auto const implPipeline = sDevice->pipelineGraphicsPool.get(draw.pipeline);
	if (!implCmd || !implPipeline) {
		return;
	}
	if (implPipeline->pipeline == VK_NULL_HANDLE) { return; }

	// bind pipeline only if it changed on this command buffer
	if (
		   implCmd->boundPipeline.id != draw.pipeline.id
		|| implCmd->bindPoint != VK_PIPELINE_BIND_POINT_GRAPHICS
	) {
		vkCmdBindPipeline(
			implCmd->cmd,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			implPipeline->pipeline
		);
		vkCmdBindDescriptorSets(
			implCmd->cmd,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			sDevice->pipelineLayoutUniversal,
			0u, 1u, &sBindlessSet,
			0u, nullptr
		);
		implCmd->boundPipeline = draw.pipeline;
		implCmd->bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	}

	// per-draw push goes to the [128,256) window
	if (draw.pushconstant.size() > 0) {
		vkCmdPushConstants(
			implCmd->cmd,
			sDevice->pipelineLayoutUniversal,
			VK_SHADER_STAGE_ALL,
			/*offset=*/ 128,
			(u32)draw.pushconstant.size(),
			draw.pushconstant.ptr()
		);
	}

	pfnVkCmdDrawMeshTasksEXT(
		implCmd->cmd,
		/*groupCountX=*/ draw.vertexCount,
		/*groupCountY=*/ draw.instanceCount,
		/*groupCountZ=*/ 1
	);
}

void vkof::cmd_dispatch(CmdDispatch const & dispatch)
{
	auto const implCmd = sDevice->commandBufferPool.get(dispatch.cmd);
	auto const implPipeline = (
		sDevice->pipelineComputePool.get(dispatch.pipeline)
	);
	if (!implCmd || !implPipeline) {
		printf("[vkof] cmd_dispatch: invalid command buffer or pipeline\n");
		return;
	}
	if (implPipeline->pipeline == VK_NULL_HANDLE) { return; }

	if (
		   implCmd->boundPipeline.id != dispatch.pipeline.id
		|| implCmd->bindPoint != VK_PIPELINE_BIND_POINT_COMPUTE
	) {
		vkCmdBindPipeline(
			implCmd->cmd,
			VK_PIPELINE_BIND_POINT_COMPUTE,
			implPipeline->pipeline
		);
		vkCmdBindDescriptorSets(
			implCmd->cmd,
			VK_PIPELINE_BIND_POINT_COMPUTE,
			sDevice->pipelineLayoutUniversal,
			0u, 1u, &sBindlessSet,
			0u, nullptr
		);
		implCmd->boundPipeline = dispatch.pipeline;
		implCmd->bindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
	}

	if (dispatch.pushconstant.size() > 0) {
		vkCmdPushConstants(
			implCmd->cmd,
			sDevice->pipelineLayoutUniversal,
			VK_SHADER_STAGE_ALL,
			/*offset=*/ 128,
			(u32)dispatch.pushconstant.size(),
			dispatch.pushconstant.ptr()
		);
	}

	u32 const gx = (
		(dispatch.invocationCount.x + dispatch.threadgroupSize.x - 1u)
		/ dispatch.threadgroupSize.x
	);
	u32 const gy = (
		(dispatch.invocationCount.y + dispatch.threadgroupSize.y - 1u)
		/ dispatch.threadgroupSize.y
	);
	u32 const gz = (
		(dispatch.invocationCount.z + dispatch.threadgroupSize.z - 1u)
		/ dispatch.threadgroupSize.z
	);
	vkCmdDispatch(implCmd->cmd, gx, gy, gz);
}

static bool resolve_image(
	vkof::Image const & image,
	VkImage & outImage,
	u32 & outWidth,
	u32 & outHeight
) {
	if (image.id == (u64)-1) {
		outImage = sDevice->swapchainImages[sDevice->swapchainImageIndex];
		outWidth = sDevice->swapchain.extent.width;
		outHeight = sDevice->swapchain.extent.height;
		return true;
	}
	if (auto const impl = sDevice->imagePool.get(image)) {
		outImage = impl->image;
		outWidth = impl->width;
		outHeight = impl->height;
		return true;
	}
	return false;
}

void vkof::cmd_copy_image(CmdCopyImage const & copy)
{
	auto const implCmd = sDevice->commandBufferPool.get(copy.cmd);
	if (!implCmd) {
		return;
	}

	VkImage srcImage;
	VkImage dstImage;
	u32 srcWidth, srcHeight, dstWidth, dstHeight;
	if (
		   !resolve_image(copy.srcImage, srcImage, srcWidth, srcHeight)
		|| !resolve_image(copy.dstImage, dstImage, dstWidth, dstHeight)
	) {
		return;
	}

	VkImageCopy const region = {
		.srcSubresource = VkImageSubresourceLayers {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.mipLevel = copy.srcBaseMipLevel,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
		.srcOffset = VkOffset3D {
			.x = (i32)copy.srcOffsetX,
			.y = (i32)copy.srcOffsetY,
			.z = 0,
		},
		.dstSubresource = VkImageSubresourceLayers {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.mipLevel = copy.dstBaseMipLevel,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
		.dstOffset = VkOffset3D {
			.x = (i32)copy.dstOffsetX,
			.y = (i32)copy.dstOffsetY,
			.z = 0,
		},
		.extent = VkExtent3D {
			.width = copy.width,
			.height = copy.height,
			.depth = 1,
		},
	};

	// the graph leaves images in GENERAL at pass boundaries, so both ends
	// are GENERAL here; no transition needed inside the copy.
	vkCmdCopyImage(
		implCmd->cmd,
		srcImage,
		VK_IMAGE_LAYOUT_GENERAL,
		dstImage,
		VK_IMAGE_LAYOUT_GENERAL,
		1,
		&region
	);
}

// -----------------------------------------------------------------------------
// -- render graph
// -----------------------------------------------------------------------------

GLFWwindow * vkof::window()
{
	return sDevice->window;
}

void vkof::imgui_begin()
{
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	sDevice->imguiFrameActive = true;
}

static void profiler_init()
{
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(sDevice->physicalDevice, &props);

	u32 qfCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(
		sDevice->physicalDevice, &qfCount, nullptr
	);
	std::vector<VkQueueFamilyProperties> qfProps(qfCount);
	vkGetPhysicalDeviceQueueFamilyProperties(
		sDevice->physicalDevice, &qfCount, qfProps.data()
	);

	auto const gfxFamilyResult = (
		sDevice->device.get_queue_index(vkb::QueueType::graphics)
	);
	if (!gfxFamilyResult.has_value()) { return; }
	u32 const gfxFamily = gfxFamilyResult.value();
	if (gfxFamily >= qfCount || qfProps[gfxFamily].timestampValidBits == 0) {
		return;
	}

	VkQueryPoolCreateInfo const ci = {
		.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.queryType = VK_QUERY_TYPE_TIMESTAMP,
		.queryCount = kMaxProfiledNodes * 2 * kFramesInFlight,
		.pipelineStatistics = 0,
	};
	if (
		vkCreateQueryPool(
			sDevice->device, &ci, nullptr, &sDevice->profiler.queryPool
		) != VK_SUCCESS
	) {
		return;
	}
	sDevice->profiler.timestampPeriodMs = (
		(double)props.limits.timestampPeriod / 1e6
	);
	sDevice->profiler.gpuSupported = true;
}

void vkof::render_graph_execute(RenderGraphExecuteInfo const & exec)
{
	sProbeMessage.clear();
	sShaderReloaded = false;

	u32 const frameSlot = sDevice->frameIndex % kFramesInFlight;
	auto & fd = sDevice->frameData[frameSlot];

	// -- wait for the previous use of this frame slot to finish
	VkAssert(
		vkWaitForFences(
			sDevice->device, 1, &fd.fence, VK_TRUE, UINT64_MAX
		)
	);
	VkAssert(vkResetFences(sDevice->device, 1, &fd.fence));

	// -- read back GPU timestamps from the previous use of this frame slot
	//    (fence above guarantees GPU is done, so results are available)
	Profiler & prof = sDevice->profiler;
	{
		u32 const prevCount = prof.slotNodeCounts[frameSlot];
		if (prof.gpuSupported && prevCount > 0) {
			std::array<u64, kMaxProfiledNodes * 2> ts {};
			VkResult const qr = vkGetQueryPoolResults(
				sDevice->device, prof.queryPool,
				frameSlot * kMaxProfiledNodes * 2, prevCount * 2,
				prevCount * 2 * sizeof(u64), ts.data(),
				sizeof(u64), VK_QUERY_RESULT_64_BIT
			);
			if (qr == VK_SUCCESS) {
				for (u32 i = 0; i < prevCount; ++i) {
					u64 const t0 = ts[i * 2 + 0];
					u64 const t1 = ts[i * 2 + 1];
					prof.timings[i].gpuMs = (double)(t1 - t0) * prof.timestampPeriodMs;
				}
			}
		}
		prof.displayCount = prevCount;
		prof.frameNodeCount = 0;
	}

	// -- acquire the next swapchain image (windowed only)
	if (sDevice->window != nullptr) {
		VkAssert(
			vkAcquireNextImageKHR(
				sDevice->device,
				sDevice->swapchain.swapchain,
				UINT64_MAX,
				fd.semaphoreAcquire,
				VK_NULL_HANDLE,
				&sDevice->swapchainImageIndex
			)
		);
	}

	VkCommandBuffer const cmd = fd.cmdGraphics;
	VkAssert(vkResetCommandBuffer(cmd, 0));
	VkCommandBufferBeginInfo const beginInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = nullptr,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = nullptr,
	};
	VkAssert(vkBeginCommandBuffer(cmd, &beginInfo));

	// -- make previous frame's persistent-resource writes visible to this frame
	{
		VkMemoryBarrier2 const crossFrameBarrier = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
			.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
		};
		VkDependencyInfo const crossFrameDep = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.pNext = nullptr,
			.dependencyFlags = 0,
			.memoryBarrierCount = 1,
			.pMemoryBarriers = &crossFrameBarrier,
			.bufferMemoryBarrierCount = 0,
			.pBufferMemoryBarriers = nullptr,
			.imageMemoryBarrierCount = 0,
			.pImageMemoryBarriers = nullptr,
		};
		vkCmdPipelineBarrier2(cmd, &crossFrameDep);
	}

	// -- reset timestamp queries for this frame slot
	if (prof.gpuSupported) {
		vkCmdResetQueryPool(
			cmd, prof.queryPool,
			frameSlot * kMaxProfiledNodes * 2, kMaxProfiledNodes * 2
		);
	}

	// -- root push constants go in the [0, 128) window
	if (exec.rootPushconstant.size() > 0) {
		vkCmdPushConstants(
			cmd,
			sDevice->pipelineLayoutUniversal,
			VK_SHADER_STAGE_ALL,
			/*offset=*/ 0,
			(u32)exec.rootPushconstant.size(),
			exec.rootPushconstant.ptr()
		);
	}

	// -- collect every unique transient image referenced by any node so we
	//    can batch all UNDEFINED->GENERAL transitions into one barrier call.
	//    the swapchain image always gets the same treatment.
	std::vector<VkImageMemoryBarrier2> initBarriers;
	{
		std::vector<u64> seen; // transientImage ids already added

		auto const addTransient = [&](vkof::TransientImage const & ti) {
			if (!ti.id) { return; }
			for (u64 id : seen) {
				if (id == ti.id) { return; }
			}
			seen.push_back(ti.id);

			vkof::Image const img = vkof::transient_image_get_image(ti);
			auto const implTex = sDevice->imagePool.get(img);
			if (!implTex) { return; }

			VkFormat const fmt = to_vk_format(
				sDevice->transientImagePool.get(ti)->format
			);
			VkImageAspectFlags const aspect = format_aspect(fmt);

			initBarriers.push_back(VkImageMemoryBarrier2 {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
				.pNext = nullptr,
				.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
				.srcAccessMask = VK_ACCESS_2_NONE,
				.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
				.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
				.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.newLayout = VK_IMAGE_LAYOUT_GENERAL,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = implTex->image,
				.subresourceRange = VkImageSubresourceRange {
					.aspectMask     = aspect,
					.baseMipLevel   = 0,
					.levelCount     = VK_REMAINING_MIP_LEVELS,
					.baseArrayLayer = 0,
					.layerCount     = 1,
				},
			});
		};

		for (auto const & nodeHandle : exec.nodes) {
			auto const node = sDevice->renderNodePool.get(nodeHandle);
			if (!node) { continue; }
			for (auto const & b : node->imageBarriers) {
				addTransient(b.transientImage);
			}
			for (auto const & att : node->colorAttachments) {
				addTransient(att.image);
			}
			if (node->depthAttachment) {
				addTransient(node->depthAttachment->image);
			}
		}

		// always transition the swapchain image UNDEFINED->GENERAL
		initBarriers.push_back(VkImageMemoryBarrier2 {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
			.srcAccessMask = VK_ACCESS_2_NONE,
			.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_GENERAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = sDevice->swapchainImages[sDevice->swapchainImageIndex],
			.subresourceRange = VkImageSubresourceRange {
				.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel   = 0,
				.levelCount     = 1,
				.baseArrayLayer = 0,
				.layerCount     = 1,
			},
		});
	}
	if (!initBarriers.empty()) {
		VkDependencyInfo const dep = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.pNext = nullptr,
			.dependencyFlags = 0,
			.memoryBarrierCount = 0,
			.pMemoryBarriers = nullptr,
			.bufferMemoryBarrierCount = 0,
			.pBufferMemoryBarriers = nullptr,
			.imageMemoryBarrierCount = (u32)initBarriers.size(),
			.pImageMemoryBarriers   = initBarriers.data(),
		};
		vkCmdPipelineBarrier2(cmd, &dep);
	}

	// -- allocate one command-buffer handle to hand to all node callbacks
	ImplCommandBuffer implCmdBuf = {
		.cmd = cmd,
		.boundPipeline = { .id = 0 },
		.bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
	};
	vkof::CommandBuffer const cmdHandle = (
		sDevice->commandBufferPool.allocate(implCmdBuf)
	);

	// -- global memory barrier emitted between nodes
	VkMemoryBarrier2 const nodeBoundaryBarrier = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
		.pNext = nullptr,
		.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		.srcAccessMask = (
			VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT
		),
		.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		.dstAccessMask = (
			VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT
		),
	};
	VkDependencyInfo const nodeBoundaryDep = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.pNext = nullptr,
		.dependencyFlags = 0,
		.memoryBarrierCount = 1,
		.pMemoryBarriers = &nodeBoundaryBarrier,
		.bufferMemoryBarrierCount = 0,
		.pBufferMemoryBarriers = nullptr,
		.imageMemoryBarrierCount = 0,
		.pImageMemoryBarriers = nullptr,
	};

	// -- execute each node in declared order
	bool firstNode = true;
	for (auto const & nodeHandle : exec.nodes) {
		auto const node = sDevice->renderNodePool.get(nodeHandle);
		if (!node) { continue; }

		u32 const nodeIdx = prof.frameNodeCount++;
		bool const canProfile = prof.gpuSupported && nodeIdx < kMaxProfiledNodes;
		auto const cpuT0 = std::chrono::steady_clock::now();

		if (canProfile) {
			vkCmdWriteTimestamp2(
				cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
				prof.queryPool, frameSlot * kMaxProfiledNodes * 2 + nodeIdx * 2
			);
		}

		// barrier between nodes (skip before the very first)
		if (!firstNode) {
			vkCmdPipelineBarrier2(cmd, &nodeBoundaryDep);
		}
		firstNode = false;

		// -- dynamic rendering setup if this node has attachments
		bool const hasColor = !node->colorAttachments.empty();
		bool const hasDepth = node->depthAttachment.has_value();
		if (hasColor || hasDepth) {
			// determine render area from the first attachment's image
			u32 renderW = sDevice->renderExtent.width;
			u32 renderH = sDevice->renderExtent.height;
			if (hasColor) {
				auto const & firstAtt = node->colorAttachments[0];
				auto const tiImpl = (
					sDevice->transientImagePool.get(firstAtt.image)
				);
				if (tiImpl) {
					renderW = (u32)(
						sDevice->renderExtent.width * tiImpl->scaleWidth
					);
					renderH = (u32)(
						sDevice->renderExtent.height * tiImpl->scaleHeight
					);
				}
			} else if (hasDepth) {
				auto const tiImpl = (
					sDevice->transientImagePool.get(
						node->depthAttachment->image
					)
				);
				if (tiImpl) {
					renderW = (u32)(
						sDevice->renderExtent.width * tiImpl->scaleWidth
					);
					renderH = (u32)(
						sDevice->renderExtent.height * tiImpl->scaleHeight
					);
				}
			}

			// sort color attachments by colorIndex so pColorAttachments
			// is in the right order.
			u32 maxColorIdx = 0;
			for (auto const & att : node->colorAttachments) {
				if (att.colorIndex > maxColorIdx) {
					maxColorIdx = att.colorIndex;
				}
			}
			std::vector<VkRenderingAttachmentInfo> colorInfos(
				maxColorIdx + 1,
				VkRenderingAttachmentInfo {
					.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
					.pNext = nullptr,
					.imageView = VK_NULL_HANDLE,
					.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
					.resolveMode = VK_RESOLVE_MODE_NONE,
					.resolveImageView = VK_NULL_HANDLE,
					.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
					.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
					.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
					.clearValue = {},
				}
			);
			for (auto const & att : node->colorAttachments) {
				vkof::Image const img = (
					vkof::transient_image_get_image(att.image)
				);
				auto const tex = sDevice->imagePool.get(img);
				if (!tex) { continue; }
				VkImageView const view = (
					att.mipLevel == 0
						? tex->imageViewFull
						: image_storage_view_for_mip(*tex, att.mipLevel)
				);
				auto const toLoadOp = [](vkof::RenderNodeLoadOp op) {
					switch (op) {
						case vkof::RenderNodeLoadOp::clear:
							return VK_ATTACHMENT_LOAD_OP_CLEAR;
						case vkof::RenderNodeLoadOp::discard:
							return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
						default:
							return VK_ATTACHMENT_LOAD_OP_LOAD;
					}
				};
				colorInfos[att.colorIndex] = VkRenderingAttachmentInfo {
					.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
					.pNext = nullptr,
					.imageView = view,
					.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
					.resolveMode = VK_RESOLVE_MODE_NONE,
					.resolveImageView = VK_NULL_HANDLE,
					.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
					.loadOp = toLoadOp(att.loadOp),
					.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
					.clearValue = VkClearValue {
						.color = {
							.float32 = {
								att.clearColor[0],
								att.clearColor[1],
								att.clearColor[2],
								att.clearColor[3],
							},
						},
					},
				};
			}

			VkRenderingAttachmentInfo depthInfo {};
			if (hasDepth) {
				auto const & datt = *node->depthAttachment;
				vkof::Image const img = (
					vkof::transient_image_get_image(datt.image)
				);
				auto const tex = sDevice->imagePool.get(img);
				VkImageView const view = (
					tex
						? (datt.mipLevel == 0
							? tex->imageViewFull
							: image_storage_view_for_mip(*tex, datt.mipLevel))
						: VK_NULL_HANDLE
				);
				auto const toLoadOp = [](vkof::RenderNodeLoadOp op) {
					switch (op) {
						case vkof::RenderNodeLoadOp::clear:
							return VK_ATTACHMENT_LOAD_OP_CLEAR;
						case vkof::RenderNodeLoadOp::discard:
							return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
						default:
							return VK_ATTACHMENT_LOAD_OP_LOAD;
					}
				};
				depthInfo = VkRenderingAttachmentInfo {
					.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
					.pNext = nullptr,
					.imageView = view,
					.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
					.resolveMode = VK_RESOLVE_MODE_NONE,
					.resolveImageView = VK_NULL_HANDLE,
					.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
					.loadOp = toLoadOp(datt.loadOp),
					.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
					.clearValue = VkClearValue {
						.depthStencil = { datt.clearDepth, 0 },
					},
				};
			}

			VkRenderingInfo const renderingInfo = {
				.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
				.pNext = nullptr,
				.flags = 0,
				.renderArea = VkRect2D {
					.offset = { 0, 0 },
					.extent = { renderW, renderH },
				},
				.layerCount = 1,
				.viewMask = 0,
				.colorAttachmentCount = (u32)colorInfos.size(),
				.pColorAttachments = colorInfos.data(),
				.pDepthAttachment = hasDepth ? &depthInfo : nullptr,
				.pStencilAttachment = nullptr,
			};
			vkCmdBeginRendering(cmd, &renderingInfo);

			// dynamic viewport and scissor to match the render area
			VkViewport const viewport = {
				.x = 0.0f, .y = 0.0f,
				.width = (f32)renderW, .height = (f32)renderH,
				.minDepth = 0.0f, .maxDepth = 1.0f,
			};
			VkRect2D const scissor = {
				.offset = { 0, 0 },
				.extent = { renderW, renderH },
			};
			vkCmdSetViewport(cmd, 0, 1, &viewport);
			vkCmdSetScissor(cmd, 0, 1, &scissor);
		}

		// -- invoke the node's recording callback
#if defined(VKOF_AFTERMATH)
		if (sDevice->pfnCmdSetCheckpointNV) {
			sDevice->pfnCmdSetCheckpointNV(
				cmd, (void *)(uintptr_t)nodeIdx
			);
		}
#endif
		if (node->callback) {
			node->callback(cmdHandle);
		}

		if (hasColor || hasDepth) {
			vkCmdEndRendering(cmd);
		}

		if (canProfile) {
			vkCmdWriteTimestamp2(
				cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
				prof.queryPool, frameSlot * kMaxProfiledNodes * 2 + nodeIdx * 2 + 1
			);
		}
		if (nodeIdx < kMaxProfiledNodes) {
			prof.timings[nodeIdx].cpuMs = std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - cpuT0
			).count();
		}
	}

	prof.slotNodeCounts[frameSlot] = prof.frameNodeCount;

	sDevice->commandBufferPool.free(cmdHandle);

	if (sDevice->window != nullptr && exec.finalImage.id != 0) {
		vkCmdPipelineBarrier2(cmd, &nodeBoundaryDep);

		vkof::Image const srcImg = vkof::transient_image_get_image(exec.finalImage);
		VkImage srcVkImage;
		u32 srcW, srcH;
		if (resolve_image(srcImg, srcVkImage, srcW, srcH)) {
			VkImage const dstVkImage = (
				sDevice->swapchainImages[sDevice->swapchainImageIndex]
			);
			u32 const copyW = sDevice->swapchain.extent.width;
			u32 const copyH = sDevice->swapchain.extent.height;
			VkImageBlit const region = {
				.srcSubresource = VkImageSubresourceLayers {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = 0,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
				.srcOffsets = {
					VkOffset3D { 0, 0, 0 },
					VkOffset3D { (i32)srcW, (i32)srcH, 1 },
				},
				.dstSubresource = VkImageSubresourceLayers {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = 0,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
				.dstOffsets = {
					VkOffset3D { 0, 0, 0 },
					VkOffset3D { (i32)copyW, (i32)copyH, 1 },
				},
			};
			vkCmdBlitImage(
				cmd,
				srcVkImage, VK_IMAGE_LAYOUT_GENERAL,
				dstVkImage, VK_IMAGE_LAYOUT_GENERAL,
				1, &region,
				VK_FILTER_NEAREST
			);
		}
	}

	if (
		sDevice->window != nullptr
		&& !sDebugVertices.empty()
		&& sDebugVertices.size() % 2u == 0u
	) {
		vkCmdPipelineBarrier2(cmd, &nodeBoundaryDep);

		// the debug pass renders to the swapchain image; the render-resolution
		//   depth target can only be attached while its extent still matches
		//   the swapchain, otherwise fall back to depth-test-free lines
		VkImageView depthView = VK_NULL_HANDLE;
		if (exec.debugDrawDepth.id != 0u) {
			vkof::Image const depthImg = (
				vkof::transient_image_get_image(exec.debugDrawDepth)
			);
			auto const depthTex = sDevice->imagePool.get(depthImg);
			if (
				depthTex
				&& depthTex->width == sDevice->swapchain.extent.width
				&& depthTex->height == sDevice->swapchain.extent.height
			) {
				depthView = depthTex->imageViewFull;
			}
		}
		bool const hasDepth = (depthView != VK_NULL_HANDLE);
		VkPipeline const debugPipeline = (
			hasDepth
				? sDebugLinePipelineWithDepth
				: sDebugLinePipelineNoDepth
		);
		if (debugPipeline != VK_NULL_HANDLE) {
			srat::slice<u8> hostMem = (
				vkof::buffer_host_address(sDebugVertexBuffer)
			);
			u64 const byteCount = (
				sDebugVertices.size() * sizeof(DebugVertex)
			);
			memcpy(hostMem.ptr(), sDebugVertices.data(), byteCount);

			u32 const lineCount = (u32)(sDebugVertices.size() / 2u);

			DebugDrawPC const debugPc = {
				.vertexVa = vkof::buffer_virtual_address(sDebugVertexBuffer),
				.viewProj = exec.debugDrawViewProj,
			};

			VkRenderingAttachmentInfo const colorAtt = {
				.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
				.pNext = nullptr,
				.imageView = (
					sDevice->swapchainImageViews[sDevice->swapchainImageIndex]
				),
				.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
				.resolveMode = VK_RESOLVE_MODE_NONE,
				.resolveImageView = VK_NULL_HANDLE,
				.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.clearValue = {},
			};
			VkRenderingAttachmentInfo const depthAtt = {
				.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
				.pNext = nullptr,
				.imageView = depthView,
				.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
				.resolveMode = VK_RESOLVE_MODE_NONE,
				.resolveImageView = VK_NULL_HANDLE,
				.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.clearValue = {},
			};

			u32 const renderW = sDevice->swapchain.extent.width;
			u32 const renderH = sDevice->swapchain.extent.height;

			VkRenderingInfo const renderingInfo = {
				.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
				.pNext = nullptr,
				.flags = 0,
				.renderArea = VkRect2D {
					.offset = { 0, 0 },
					.extent = { renderW, renderH },
				},
				.layerCount = 1,
				.viewMask = 0,
				.colorAttachmentCount = 1,
				.pColorAttachments = &colorAtt,
				.pDepthAttachment = (
					(hasDepth && depthView != VK_NULL_HANDLE) ? &depthAtt : nullptr
				),
				.pStencilAttachment = nullptr,
			};

			vkCmdBeginRendering(cmd, &renderingInfo);
			vkCmdBindPipeline(
				cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, debugPipeline
			);
			vkCmdBindDescriptorSets(
				cmd,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				sDevice->pipelineLayoutUniversal,
				0u, 1u, &sBindlessSet,
				0u, nullptr
			);
			VkViewport const viewport = {
				.x = 0.0f,
				.y = 0.0f,
				.width = (f32)renderW,
				.height = (f32)renderH,
				.minDepth = 0.0f,
				.maxDepth = 1.0f,
			};
			VkRect2D const scissor = {
				.offset = { 0, 0 },
				.extent = { renderW, renderH },
			};
			vkCmdSetViewport(cmd, 0u, 1u, &viewport);
			vkCmdSetScissor(cmd, 0u, 1u, &scissor);
			vkCmdPushConstants(
				cmd,
				sDevice->pipelineLayoutUniversal,
				VK_SHADER_STAGE_ALL,
				128u,
				(u32)sizeof(DebugDrawPC),
				&debugPc
			);
			pfnVkCmdDrawMeshTasksEXT(cmd, lineCount, 1u, 1u);
			vkCmdEndRendering(cmd);
		}
	}

	if (
		sDevice->window != nullptr
		&& !sDebugSphereBatches.empty()
		&& exec.debugDrawDepth.id != 0u
	) {
		vkCmdPipelineBarrier2(cmd, &nodeBoundaryDep);

		u32 const sphereCount = (u32)sDebugSphereData.size();
		if (sphereCount > sDebugSphereCapacity) {
			vkof::buffer_destroy(sDebugSphereBuffer);
			sDebugSphereCapacity = sphereCount * 2u;
			sDebugSphereBuffer = vkof::buffer_create({
				.byteCount = sizeof(vkof::DebugSphere) * sDebugSphereCapacity,
				.memory = vkof::BufferMemory::HostWritable,
			});
		}
		srat::slice<u8> sphereHostMem = (
			vkof::buffer_host_address(sDebugSphereBuffer)
		);
		u64 const sphereByteCount = (
			sDebugSphereData.size() * sizeof(vkof::DebugSphere)
		);
		memcpy(
			sphereHostMem.ptr(), sDebugSphereData.data(), sphereByteCount
		);
		u64 const sphereVa = (
			vkof::buffer_virtual_address(sDebugSphereBuffer)
		);

		vkof::Image const sphereDepthImg = (
			vkof::transient_image_get_image(exec.debugDrawDepth)
		);
		auto const sphereDepthTex = sDevice->imagePool.get(sphereDepthImg);
		// same swapchain-extent restriction as the debug line pass above
		VkImageView const sphereDepthView = (
			(
				sphereDepthTex
				&& sphereDepthTex->width == sDevice->swapchain.extent.width
				&& sphereDepthTex->height == sDevice->swapchain.extent.height
			)
				? sphereDepthTex->imageViewFull
				: VK_NULL_HANDLE
		);

		VkRenderingAttachmentInfo const sphereColorAtt = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.pNext = nullptr,
			.imageView = (
				sDevice->swapchainImageViews[sDevice->swapchainImageIndex]
			),
			.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
			.resolveMode = VK_RESOLVE_MODE_NONE,
			.resolveImageView = VK_NULL_HANDLE,
			.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = {},
		};
		VkRenderingAttachmentInfo const sphereDepthAtt = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.pNext = nullptr,
			.imageView = sphereDepthView,
			.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
			.resolveMode = VK_RESOLVE_MODE_NONE,
			.resolveImageView = VK_NULL_HANDLE,
			.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = {},
		};
		u32 const sphereRenderW = sDevice->swapchain.extent.width;
		u32 const sphereRenderH = sDevice->swapchain.extent.height;
		VkRenderingInfo const sphereRenderingInfo = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
			.pNext = nullptr,
			.flags = 0,
			.renderArea = VkRect2D {
				.offset = { 0, 0 },
				.extent = { sphereRenderW, sphereRenderH },
			},
			.layerCount = 1,
			.viewMask = 0,
			.colorAttachmentCount = 1,
			.pColorAttachments = &sphereColorAtt,
			.pDepthAttachment = (
				sphereDepthView != VK_NULL_HANDLE ? &sphereDepthAtt : nullptr
			),
			.pStencilAttachment = nullptr,
		};
		vkCmdBeginRendering(cmd, &sphereRenderingInfo);
		VkViewport const sphereViewport = {
			.x = 0.0f,
			.y = 0.0f,
			.width = (f32)sphereRenderW,
			.height = (f32)sphereRenderH,
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};
		VkRect2D const sphereScissor = {
			.offset = { 0, 0 },
			.extent = { sphereRenderW, sphereRenderH },
		};
		vkCmdSetViewport(cmd, 0u, 1u, &sphereViewport);
		vkCmdSetScissor(cmd, 0u, 1u, &sphereScissor);
		for (DebugSphereBatch const & batch : sDebugSphereBatches) {
			auto const batchImpl = (
				sDevice->pipelineGraphicsPool.get(batch.pipeline)
			);
			if (!batchImpl || batchImpl->pipeline == VK_NULL_HANDLE) {
				continue;
			}
			vkCmdBindPipeline(
				cmd,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				batchImpl->pipeline
			);
			vkCmdBindDescriptorSets(
				cmd,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				sDevice->pipelineLayoutUniversal,
				0u, 1u, &sBindlessSet,
				0u, nullptr
			);
			DebugSpherePC const spherePc = {
				.sphereVa = sphereVa,
				.viewProj = exec.debugDrawViewProj,
				.startIdx = batch.startIdx,
			};
			vkCmdPushConstants(
				cmd,
				sDevice->pipelineLayoutUniversal,
				VK_SHADER_STAGE_ALL,
				128u,
				(u32)sizeof(DebugSpherePC),
				&spherePc
			);
			pfnVkCmdDrawMeshTasksEXT(cmd, batch.count, 1u, 1u);
		}
		vkCmdEndRendering(cmd);
	}

	if (sDevice->window != nullptr && sDevice->imguiFrameActive) {
		sDevice->imguiFrameActive = false;
		vkCmdPipelineBarrier2(cmd, &nodeBoundaryDep);

		// -- profiler window
		if (prof.displayCount > 0) {
			ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
			ImGui::Begin("profiler", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
			ImGui::Text("%-6s  %8s  %8s", "node", "cpu ms", "gpu ms");
			ImGui::Separator();
			for (u32 i = 0; i < prof.displayCount; ++i) {
				NodeTiming const & t = prof.timings[i];
				ImGui::Text("node %-2u  %8.3f  %8.3f", i, t.cpuMs, t.gpuMs);
			}
			ImGui::End();
		}

		// -- vma memory dashboard
		{
			VkPhysicalDeviceMemoryProperties memProps;
			vkGetPhysicalDeviceMemoryProperties(
				sDevice->physicalDevice, &memProps
			);
			VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
			vmaGetHeapBudgets(sDevice->allocator, budgets);

			ImGui::SetNextWindowPos(ImVec2(10.0f, 150.0f), ImGuiCond_FirstUseEver);
			ImGui::Begin("memory", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
			for (u32 i = 0; i < memProps.memoryHeapCount; ++i) {
				VmaBudget const & b = budgets[i];
				bool const isDevice = (
					(
						memProps.memoryHeaps[i].flags
						& VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
					) != 0
				);
				f32 const usedMb = (f32)b.usage / (1024.0f * 1024.0f);
				f32 const budgetMb = (f32)b.budget / (1024.0f * 1024.0f);
				f32 const fraction = budgetMb > 0.0f ? usedMb / budgetMb : 0.0f;
				ImGui::Text("heap %u  %s", i, isDevice ? "device" : "host");
				ImGui::ProgressBar(fraction, ImVec2(180.0f, 0.0f));
				ImGui::SameLine();
				ImGui::Text("%.0f / %.0f MB", usedMb, budgetMb);
				ImGui::Text(
					"  allocs %u  blocks %u",
					b.statistics.allocationCount, b.statistics.blockCount
				);
			}
			ImGui::End();
		}

		// -- shader error overlay
		if (!sLastCompileError.empty()) {
			ImGui::SetNextWindowPos(ImVec2(10.0f, 400.0f), ImGuiCond_FirstUseEver);
			ImGui::SetNextWindowSize(ImVec2(620.0f, 260.0f), ImGuiCond_FirstUseEver);
			ImGui::PushStyleColor(
				ImGuiCol_TitleBg, ImVec4(0.5f, 0.05f, 0.05f, 1.0f)
			);
			ImGui::PushStyleColor(
				ImGuiCol_TitleBgActive, ImVec4(0.7f, 0.05f, 0.05f, 1.0f)
			);
			ImGui::Begin("shader error", nullptr, 0);
			ImGui::PopStyleColor(2);
			ImGui::TextUnformatted(sLastCompileError.c_str());
			ImGui::End();
		}

		ImGui::Render();
		ImDrawData * const imguiDrawData = ImGui::GetDrawData();

		VkRenderingAttachmentInfo const imguiColorAtt = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.pNext = nullptr,
			.imageView = sDevice->swapchainImageViews[sDevice->swapchainImageIndex],
			.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
			.resolveMode = VK_RESOLVE_MODE_NONE,
			.resolveImageView = VK_NULL_HANDLE,
			.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = {},
		};
		VkRenderingInfo const imguiRenderingInfo = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
			.pNext = nullptr,
			.flags = 0,
			.renderArea = VkRect2D {
				.offset = { 0, 0 },
				.extent = {
					sDevice->swapchain.extent.width,
					sDevice->swapchain.extent.height,
				},
			},
			.layerCount = 1,
			.viewMask = 0,
			.colorAttachmentCount = 1,
			.pColorAttachments = &imguiColorAtt,
			.pDepthAttachment = nullptr,
			.pStencilAttachment = nullptr,
		};
		vkCmdBeginRendering(cmd, &imguiRenderingInfo);
		if (imguiDrawData) {
			ImGui_ImplVulkan_RenderDrawData(imguiDrawData, cmd);
		}
		vkCmdEndRendering(cmd);
	}

	// -- transition swapchain GENERAL -> PRESENT_SRC (windowed only)
	if (sDevice->window != nullptr) {
		VkImageMemoryBarrier2 const presentBarrier = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
			.dstStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
			.dstAccessMask = VK_ACCESS_2_NONE,
			.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
			.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = sDevice->swapchainImages[sDevice->swapchainImageIndex],
			.subresourceRange = VkImageSubresourceRange {
				.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel   = 0,
				.levelCount     = 1,
				.baseArrayLayer = 0,
				.layerCount     = 1,
			},
		};
		VkDependencyInfo const presentDep = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.pNext = nullptr,
			.dependencyFlags = 0,
			.memoryBarrierCount = 0,
			.pMemoryBarriers = nullptr,
			.bufferMemoryBarrierCount = 0,
			.pBufferMemoryBarriers = nullptr,
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &presentBarrier,
		};
		vkCmdPipelineBarrier2(cmd, &presentDep);
	}

	VkAssert(vkEndCommandBuffer(cmd));

	VkCommandBufferSubmitInfo const cmdSubmit = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
		.pNext = nullptr,
		.commandBuffer = cmd,
		.deviceMask = 0,
	};
	if (sDevice->window != nullptr) {
		// windowed: wait on acquire semaphore, signal render semaphore + fence
		VkSemaphoreSubmitInfo const waitInfo = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
			.pNext = nullptr,
			.semaphore = fd.semaphoreAcquire,
			.value = 0,
			.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
			.deviceIndex = 0,
		};
		VkSemaphore const renderSem = (
			sDevice->semaphoresRender[sDevice->swapchainImageIndex]
		);
		VkSemaphoreSubmitInfo const signalInfo = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
			.pNext = nullptr,
			.semaphore = renderSem,
			.value = 0,
			.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
			.deviceIndex = 0,
		};
		VkSubmitInfo2 const submitInfo = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
			.pNext = nullptr,
			.flags = 0,
			.waitSemaphoreInfoCount   = 1,
			.pWaitSemaphoreInfos      = &waitInfo,
			.commandBufferInfoCount   = 1,
			.pCommandBufferInfos      = &cmdSubmit,
			.signalSemaphoreInfoCount = 1,
			.pSignalSemaphoreInfos    = &signalInfo,
		};
		VkAssert(
			vkQueueSubmit2(
				sDevice->queueGraphics, 1, &submitInfo, fd.fence
			)
		);
		VkPresentInfoKHR const presentInfo = {
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.pNext = nullptr,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &renderSem,
			.swapchainCount = 1,
			.pSwapchains = &sDevice->swapchain.swapchain,
			.pImageIndices = &sDevice->swapchainImageIndex,
			.pResults = nullptr,
		};
		vkQueuePresentKHR(sDevice->queueGraphics, &presentInfo);
	} else {
		// headless: no acquire/present semaphores, submit with fence only
		VkSubmitInfo2 const submitInfo = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
			.pNext = nullptr,
			.flags = 0,
			.waitSemaphoreInfoCount   = 0,
			.pWaitSemaphoreInfos      = nullptr,
			.commandBufferInfoCount   = 1,
			.pCommandBufferInfos      = &cmdSubmit,
			.signalSemaphoreInfoCount = 0,
			.pSignalSemaphoreInfos    = nullptr,
		};
		VkAssert(
			vkQueueSubmit2(
				sDevice->queueGraphics, 1, &submitInfo, fd.fence
			)
		);
	}

	sDebugVertices.clear();
	sDebugSphereData.clear();
	sDebugSphereBatches.clear();
	++sDevice->frameIndex;
	pipeline_hot_reload();
}

void vkof::screenshot(
	vkof::TransientImage const & image,
	char const * const path
) {
	srat::profile_tick pt;
	vkof::device_wait_idle();
	pt.tick("device_wait_idle");

	vkof::Image const img = vkof::transient_image_get_image(image);
	VkImage vkImg;
	u32 w, h;
	if (!resolve_image(img, vkImg, w, h)) { return; }

	u64 const byteCount = (u64)w * h * 4u;

	VkBuffer stagingBuf;
	VmaAllocation stagingAlloc;
	VmaAllocationInfo allocInfo {};
	{
		VkBufferCreateInfo const bci = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.size = byteCount,
			.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.queueFamilyIndexCount = 0,
			.pQueueFamilyIndices = nullptr,
		};
		VmaAllocationCreateInfo const aci = {
			.flags = (
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
				| VMA_ALLOCATION_CREATE_MAPPED_BIT
			),
			.usage = VMA_MEMORY_USAGE_AUTO,
			.requiredFlags = 0,
			.preferredFlags = 0,
			.memoryTypeBits = 0,
			.pool = nullptr,
			.pUserData = nullptr,
			.priority = 0.0f,
		};
		VkAssert(vmaCreateBuffer(
			sDevice->allocator, &bci, &aci, &stagingBuf, &stagingAlloc, &allocInfo
		));
	}
	pt.tick("staging alloc");

	VkCommandPool const pool = sDevice->commandPoolGraphics;
	VkCommandBuffer cmd;
	{
		VkCommandBufferAllocateInfo const cmdAlloc = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.pNext = nullptr,
			.commandPool = pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1u,
		};
		VkAssert(vkAllocateCommandBuffers(sDevice->device, &cmdAlloc, &cmd));
	}

	{
		VkCommandBufferBeginInfo const beginInfo = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.pNext = nullptr,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
			.pInheritanceInfo = nullptr,
		};
		VkAssert(vkBeginCommandBuffer(cmd, &beginInfo));
	}

	VkImageSubresourceRange const fullImage = {
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.baseMipLevel = 0u,
		.levelCount = 1u,
		.baseArrayLayer = 0u,
		.layerCount = 1u,
	};

	{
		VkImageMemoryBarrier2 const barrier = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
			.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
			.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = vkImg,
			.subresourceRange = fullImage,
		};
		VkDependencyInfo const dep = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.pNext = nullptr,
			.dependencyFlags = 0,
			.memoryBarrierCount = 0,
			.pMemoryBarriers = nullptr,
			.bufferMemoryBarrierCount = 0,
			.pBufferMemoryBarriers = nullptr,
			.imageMemoryBarrierCount = 1u,
			.pImageMemoryBarriers = &barrier,
		};
		vkCmdPipelineBarrier2(cmd, &dep);
	}

	{
		VkBufferImageCopy const region = {
			.bufferOffset = 0u,
			.bufferRowLength = 0u,
			.bufferImageHeight = 0u,
			.imageSubresource = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.mipLevel = 0u,
				.baseArrayLayer = 0u,
				.layerCount = 1u,
			},
			.imageOffset = { 0, 0, 0 },
			.imageExtent = { w, h, 1u },
		};
		vkCmdCopyImageToBuffer(
			cmd, vkImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			stagingBuf, 1u, &region
		);
	}

	{
		VkImageMemoryBarrier2 const barrier = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
			.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_GENERAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = vkImg,
			.subresourceRange = fullImage,
		};
		VkDependencyInfo const dep = {
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.pNext = nullptr,
			.dependencyFlags = 0,
			.memoryBarrierCount = 0,
			.pMemoryBarriers = nullptr,
			.bufferMemoryBarrierCount = 0,
			.pBufferMemoryBarriers = nullptr,
			.imageMemoryBarrierCount = 1u,
			.pImageMemoryBarriers = &barrier,
		};
		vkCmdPipelineBarrier2(cmd, &dep);
	}

	VkAssert(vkEndCommandBuffer(cmd));

	VkFence fence;
	{
		VkFenceCreateInfo const fci = {
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
		};
		VkAssert(vkCreateFence(sDevice->device, &fci, nullptr, &fence));
	}

	{
		VkCommandBufferSubmitInfo const cmdSubmit = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
			.pNext = nullptr,
			.commandBuffer = cmd,
			.deviceMask = 0,
		};
		VkSubmitInfo2 const submitInfo = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
			.pNext = nullptr,
			.flags = 0,
			.waitSemaphoreInfoCount = 0,
			.pWaitSemaphoreInfos = nullptr,
			.commandBufferInfoCount = 1u,
			.pCommandBufferInfos = &cmdSubmit,
			.signalSemaphoreInfoCount = 0,
			.pSignalSemaphoreInfos = nullptr,
		};
		VkAssert(vkQueueSubmit2(sDevice->queueGraphics, 1u, &submitInfo, fence));
	}
	pt.tick("cmd submit");

	VkAssert(vkWaitForFences(sDevice->device, 1u, &fence, VK_TRUE, UINT64_MAX));
	pt.tick("fence wait (gpu copy)");

	vkDestroyFence(sDevice->device, fence, nullptr);
	vkFreeCommandBuffers(sDevice->device, pool, 1u, &cmd);

	vkof_write_png_uncompressed(
		path, (int)w, (int)h, 4, allocInfo.pMappedData, (int)(w * 4u)
	);
	pt.tick("stbi_write_png");

	vmaDestroyBuffer(sDevice->allocator, stagingBuf, stagingAlloc);
}

// ----------------------------------------------------------------------------
// -- acceleration structures
// ----------------------------------------------------------------------------

namespace {

static VkBuffer as_alloc_buffer(
	VkDeviceSize const byteCount,
	VkBufferUsageFlags const usage,
	VmaMemoryUsage const memUsage,
	VmaAllocationCreateFlags const extraFlags,
	VmaAllocation & outAlloc,
	VmaAllocationInfo * outInfo = nullptr
) {
	VkBufferCreateInfo const bci = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.size = byteCount,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = nullptr,
	};
	VmaAllocationCreateInfo const aci = {
		.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT | extraFlags,
		.usage = memUsage,
		.requiredFlags = 0,
		.preferredFlags = 0,
		.memoryTypeBits = 0,
		.pool = nullptr,
		.pUserData = nullptr,
		.priority = 0.0f,
	};
	VkBuffer buf;
	VkAssert(
		vmaCreateBuffer(
			/*allocator=*/sDevice->allocator,
			&bci,
			&aci,
			/*pBuffer=*/&buf,
			/*pAllocation=*/&outAlloc,
			/*pAllocationInfo=*/outInfo
		)
	);
	return buf;
}

static u64 as_buffer_va(VkBuffer const buf) {
	VkBufferDeviceAddressInfo const addrInfo = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		.pNext = nullptr,
		.buffer = buf,
	};
	return vkGetBufferDeviceAddress(sDevice->device, &addrInfo);
}

static VkCommandBuffer as_begin_oneshot() {
	VkCommandBufferAllocateInfo const cbai = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = nullptr,
		.commandPool = sDevice->commandPoolGraphics,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1u,
	};
	VkCommandBuffer cmd;
	VkAssert(vkAllocateCommandBuffers(sDevice->device, &cbai, &cmd));
	VkCommandBufferBeginInfo const beginInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = nullptr,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = nullptr,
	};
	VkAssert(vkBeginCommandBuffer(cmd, &beginInfo));
	return cmd;
}

static void as_submit_oneshot(VkCommandBuffer const cmd) {
	VkAssert(vkEndCommandBuffer(cmd));
	VkSubmitInfo const submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = nullptr,
		.waitSemaphoreCount = 0u,
		.pWaitSemaphores = nullptr,
		.pWaitDstStageMask = nullptr,
		.commandBufferCount = 1u,
		.pCommandBuffers = &cmd,
		.signalSemaphoreCount = 0u,
		.pSignalSemaphores = nullptr,
	};
	VkAssert(
		vkQueueSubmit(
			/*queue=*/sDevice->queueGraphics,
			/*submitCount=*/1u,
			&submitInfo,
			/*fence=*/VK_NULL_HANDLE
		)
	);
	VkAssert(vkQueueWaitIdle(sDevice->queueGraphics));
	vkFreeCommandBuffers(
		/*device=*/sDevice->device,
		/*commandPool=*/sDevice->commandPoolGraphics,
		/*commandBufferCount=*/1u,
		&cmd
	);
}

} // namespace

vkof::AccelerationStructureBlas vkof::blas_create(BlasCreateInfo const & ci) {
	// -- size query
	VkAccelerationStructureGeometryKHR const geom = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		.pNext = nullptr,
		.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
		.geometry = {
			.triangles = {
				.sType = (
					VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR
				),
				.pNext = nullptr,
				.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
				.vertexData = { .deviceAddress = ci.positionVa },
				.vertexStride = sizeof(f32v3),
				.maxVertex = ci.vertexCount - 1u,
				.indexType = VK_INDEX_TYPE_UINT32,
				.indexData = { .deviceAddress = ci.indexVa },
				.transformData = { .deviceAddress = 0u },
			},
		},
		.flags = (
			ci.isOpaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : (VkGeometryFlagBitsKHR)0u
		),
	};
	VkAccelerationStructureBuildGeometryInfoKHR const sizeQueryInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		.pNext = nullptr,
		.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
		.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
		.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
		.srcAccelerationStructure = VK_NULL_HANDLE,
		.dstAccelerationStructure = VK_NULL_HANDLE,
		.geometryCount = 1u,
		.pGeometries = &geom,
		.ppGeometries = nullptr,
		.scratchData = { .deviceAddress = 0u },
	};
	VkAccelerationStructureBuildSizesInfoKHR sizes = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
		.pNext = nullptr,
		.accelerationStructureSize = 0u,
		.updateScratchSize = 0u,
		.buildScratchSize = 0u,
	};
	sDevice->pfnGetAccelerationStructureBuildSizesKHR(
		/*device=*/sDevice->device,
		/*buildType=*/VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&sizeQueryInfo,
		/*pMaxPrimitiveCounts=*/&ci.triangleCount,
		&sizes
	);

	// -- backing buffer + AS handle
	ImplAccelerationStructureBlas impl = {};
	impl.backingBuffer = (
		as_alloc_buffer(
			/*byteCount=*/sizes.accelerationStructureSize,
			/*usage=*/(
				VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
				| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
			),
			/*memUsage=*/VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
			/*extraFlags=*/0,
			impl.backingAlloc
		)
	);
	VkAccelerationStructureCreateInfoKHR const asCi = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
		.pNext = nullptr,
		.createFlags = 0,
		.buffer = impl.backingBuffer,
		.offset = 0u,
		.size = sizes.accelerationStructureSize,
		.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
		.deviceAddress = 0u,
	};
	VkAssert(
		sDevice->pfnCreateAccelerationStructureKHR(
			sDevice->device, &asCi, /*pAllocator=*/nullptr, &impl.handle
		)
	);
	VkAccelerationStructureDeviceAddressInfoKHR const asAddrInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
		.pNext = nullptr,
		.accelerationStructure = impl.handle,
	};
	impl.deviceAddress = (
		sDevice->pfnGetAccelerationStructureDeviceAddressKHR(
			sDevice->device,
			&asAddrInfo
		)
	);

	// -- scratch buffer (temporary, freed after build)
	VmaAllocation scratchAlloc;
	VkBuffer const scratchBuffer = as_alloc_buffer(
		/*byteCount=*/sizes.buildScratchSize,
		/*usage=*/(
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
			| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		),
		/*memUsage=*/VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
		/*extraFlags=*/0,
		scratchAlloc
	);

	// -- build
	VkAccelerationStructureBuildGeometryInfoKHR const buildInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		.pNext = nullptr,
		.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
		.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
		.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
		.srcAccelerationStructure = VK_NULL_HANDLE,
		.dstAccelerationStructure = impl.handle,
		.geometryCount = 1u,
		.pGeometries = &geom,
		.ppGeometries = nullptr,
		.scratchData = { .deviceAddress = as_buffer_va(scratchBuffer) },
	};
	VkAccelerationStructureBuildRangeInfoKHR const rangeInfo = {
		.primitiveCount = ci.triangleCount,
		.primitiveOffset = 0u,
		.firstVertex = 0u,
		.transformOffset = 0u,
	};
	VkAccelerationStructureBuildRangeInfoKHR const * const pRange = &rangeInfo;
	VkCommandBuffer const cmd = as_begin_oneshot();
	sDevice->pfnCmdBuildAccelerationStructuresKHR(
		/*commandBuffer=*/cmd,
		/*infoCount=*/1u,
		&buildInfo,
		&pRange
	);
	as_submit_oneshot(cmd);

	vmaDestroyBuffer(sDevice->allocator, scratchBuffer, scratchAlloc);
	return sDevice->blasPool.allocate(impl);
}

void vkof::blas_destroy(AccelerationStructureBlas const & blas) {
	ImplAccelerationStructureBlas const * const impl = sDevice->blasPool.get(blas);
	if (!impl) { return; }
	sDevice->pfnDestroyAccelerationStructureKHR(
		sDevice->device, impl->handle, /*pAllocator=*/nullptr
	);
	vmaDestroyBuffer(sDevice->allocator, impl->backingBuffer, impl->backingAlloc);
	sDevice->blasPool.free(blas);
}

vkof::AccelerationStructureTlas vkof::tlas_create(TlasCreateInfo const & ci) {
	// -- size query
	VkAccelerationStructureGeometryKHR const geom = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		.pNext = nullptr,
		.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
		.geometry = {
			.instances = {
				.sType = (
					VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR
				),
				.pNext = nullptr,
				.arrayOfPointers = VK_FALSE,
				.data = { .deviceAddress = 0u },
			},
		},
		.flags = 0,
	};
	VkAccelerationStructureBuildGeometryInfoKHR const sizeQueryInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		.pNext = nullptr,
		.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
		.flags = (
			VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
			| VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
		),
		.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
		.srcAccelerationStructure = VK_NULL_HANDLE,
		.dstAccelerationStructure = VK_NULL_HANDLE,
		.geometryCount = 1u,
		.pGeometries = &geom,
		.ppGeometries = nullptr,
		.scratchData = { .deviceAddress = 0u },
	};
	VkAccelerationStructureBuildSizesInfoKHR sizes = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
		.pNext = nullptr,
		.accelerationStructureSize = 0u,
		.updateScratchSize = 0u,
		.buildScratchSize = 0u,
	};
	sDevice->pfnGetAccelerationStructureBuildSizesKHR(
		/*device=*/sDevice->device,
		/*buildType=*/VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&sizeQueryInfo,
		/*pMaxPrimitiveCounts=*/&ci.maxInstances,
		&sizes
	);

	// -- backing buffer + AS handle
	ImplAccelerationStructureTlas impl = {};
	impl.maxInstances = ci.maxInstances;
	impl.backingBuffer = as_alloc_buffer(
		/*byteCount=*/sizes.accelerationStructureSize,
		/*usage=*/(
			VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
			| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		),
		/*memUsage=*/VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
		/*extraFlags=*/0,
		impl.backingAlloc
	);
	VkAccelerationStructureCreateInfoKHR const asCi = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
		.pNext = nullptr,
		.createFlags = 0,
		.buffer = impl.backingBuffer,
		.offset = 0u,
		.size = sizes.accelerationStructureSize,
		.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
		.deviceAddress = 0u,
	};
	VkAssert(
		sDevice->pfnCreateAccelerationStructureKHR(
			sDevice->device, &asCi, /*pAllocator=*/nullptr, &impl.handle
		)
	);
	VkAccelerationStructureDeviceAddressInfoKHR const asAddrInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
		.pNext = nullptr,
		.accelerationStructure = impl.handle,
	};
	impl.deviceAddress = (
		sDevice->pfnGetAccelerationStructureDeviceAddressKHR(
			sDevice->device,
			&asAddrInfo
		)
	);

	// -- scratch buffer (persistent, reused each tlas_build)
	impl.scratchBuffer = as_alloc_buffer(
		/*byteCount=*/sizes.buildScratchSize,
		/*usage=*/(
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
			| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		),
		/*memUsage=*/VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
		/*extraFlags=*/0,
		impl.scratchAlloc
	);
	impl.scratchVa = as_buffer_va(impl.scratchBuffer);

	// -- host-visible instance buffer
	VmaAllocationInfo instanceInfo = {};
	impl.instanceBuffer = as_alloc_buffer(
		/*byteCount=*/ci.maxInstances * sizeof(VkAccelerationStructureInstanceKHR),
		/*usage=*/(
			VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
			| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
		),
		/*memUsage=*/VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
		/*extraFlags=*/(
			VMA_ALLOCATION_CREATE_MAPPED_BIT
			| VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
		),
		impl.instanceAlloc,
		&instanceInfo
	);
	impl.instanceMapped = instanceInfo.pMappedData;
	impl.instanceVa = as_buffer_va(impl.instanceBuffer);

	return sDevice->tlasPool.allocate(impl);
}

void vkof::tlas_build(
	CommandBuffer const & cmd,
	AccelerationStructureTlas const & tlas,
	srat::slice<TlasInstance const> const & instances
) {
	ImplCommandBuffer const * const implCmd = sDevice->commandBufferPool.get(cmd);
	ImplAccelerationStructureTlas const * const impl = sDevice->tlasPool.get(tlas);
	if (!implCmd || !impl) { return; }
	u32 const count = (u32)instances.size();
	SRAT_ASSERT(count <= impl->maxInstances);

	// -- write instance buffer
	auto * const dst = (
		reinterpret_cast<VkAccelerationStructureInstanceKHR *>(impl->instanceMapped)
	);
	for (u32 i = 0u; i < count; ++i) {
		TlasInstance const & src = instances[i];
		ImplAccelerationStructureBlas const * const blasImpl = (
			sDevice->blasPool.get(src.blas)
		);
		SRAT_ASSERT(blasImpl);
		// f32m44 is column-major m[col*4+row]; VkTransformMatrixKHR is row-major
		VkTransformMatrixKHR xform = {};
		for (u32 r = 0u; r < 3u; ++r)
		for (u32 c = 0u; c < 4u; ++c) {
			xform.matrix[r][c] = src.transform.m[c * 4u + r];
		}
		dst[i] = VkAccelerationStructureInstanceKHR {
			.transform = xform,
			.instanceCustomIndex = src.instanceCustomIndex & 0xFFFFFFu,
			.mask = src.rayMask & 0xFFu,
			.instanceShaderBindingTableRecordOffset = 0u,
			.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
			.accelerationStructureReference = blasImpl->deviceAddress,
		};
	}

	// -- build command
	VkAccelerationStructureGeometryKHR const buildGeom = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		.pNext = nullptr,
		.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
		.geometry = {
			.instances = {
				.sType = (
					VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR
				),
				.pNext = nullptr,
				.arrayOfPointers = VK_FALSE,
				.data = { .deviceAddress = impl->instanceVa },
			},
		},
		.flags = 0,
	};
	VkAccelerationStructureBuildGeometryInfoKHR const buildInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		.pNext = nullptr,
		.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
		.flags = (
			VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
			| VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
		),
		.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
		.srcAccelerationStructure = VK_NULL_HANDLE,
		.dstAccelerationStructure = impl->handle,
		.geometryCount = 1u,
		.pGeometries = &buildGeom,
		.ppGeometries = nullptr,
		.scratchData = { .deviceAddress = impl->scratchVa },
	};
	VkAccelerationStructureBuildRangeInfoKHR const rangeInfo = {
		.primitiveCount = count,
		.primitiveOffset = 0u,
		.firstVertex = 0u,
		.transformOffset = 0u,
	};
	VkAccelerationStructureBuildRangeInfoKHR const * const pRange = &rangeInfo;
	sDevice->pfnCmdBuildAccelerationStructuresKHR(
		/*commandBuffer=*/implCmd->cmd,
		/*infoCount=*/1u,
		&buildInfo,
		&pRange
	);

	// -- barrier: AS build write -> compute shader read
	VkMemoryBarrier2 const barrier = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
		.pNext = nullptr,
		.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
		.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
		.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
	};
	VkDependencyInfo const depInfo = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.pNext = nullptr,
		.dependencyFlags = 0,
		.memoryBarrierCount = 1u,
		.pMemoryBarriers = &barrier,
		.bufferMemoryBarrierCount = 0u,
		.pBufferMemoryBarriers = nullptr,
		.imageMemoryBarrierCount = 0u,
		.pImageMemoryBarriers = nullptr,
	};
	vkCmdPipelineBarrier2(implCmd->cmd, &depInfo);
}

void vkof::tlas_destroy(AccelerationStructureTlas const & tlas) {
	ImplAccelerationStructureTlas const * const impl = sDevice->tlasPool.get(tlas);
	if (!impl) { return; }
	sDevice->pfnDestroyAccelerationStructureKHR(
		sDevice->device, impl->handle, /*pAllocator=*/nullptr
	);
	vmaDestroyBuffer(sDevice->allocator, impl->backingBuffer, impl->backingAlloc);
	vmaDestroyBuffer(sDevice->allocator, impl->scratchBuffer, impl->scratchAlloc);
	vmaDestroyBuffer(
		sDevice->allocator, impl->instanceBuffer, impl->instanceAlloc
	);
	sDevice->tlasPool.free(tlas);
}

u64 vkof::acceleration_structure_device_address(
	AccelerationStructureTlas const & as
) {
	ImplAccelerationStructureTlas const * const impl = sDevice->tlasPool.get(as);
	if (!impl) { return 0u; }
	return impl->deviceAddress;
}

void vkof::acceleration_structure_set_tlas(
	AccelerationStructureTlas const & as
) {
	ImplAccelerationStructureTlas const * const impl = sDevice->tlasPool.get(as);
	if (!impl) { return; }
	VkWriteDescriptorSetAccelerationStructureKHR const asWrite = {
		.sType = (
			VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR
		),
		.pNext = nullptr,
		.accelerationStructureCount = 1u,
		.pAccelerationStructures = &impl->handle,
	};
	VkWriteDescriptorSet const write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.pNext = &asWrite,
		.dstSet = sBindlessSet,
		.dstBinding = 3u,
		.dstArrayElement = 0u,
		.descriptorCount = 1u,
		.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
		.pImageInfo = nullptr,
		.pBufferInfo = nullptr,
		.pTexelBufferView = nullptr,
	};
	vkUpdateDescriptorSets(sDevice->device, 1u, &write, 0u, nullptr);
}
