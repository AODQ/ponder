#pragma once

#include <cstdint>
#include <source_location>

#include <srat/core-array.hpp>
#include <srat/core-math.hpp>
#include <srat/core-types.hpp>

#include <functional>
#include <imgui.h>

struct GLFWwindow;

namespace vkof
{
	enum struct ImageFormat {
		none,
		r32ui,
		r8g8b8a8_unorm,
		r8g8b8a8_srgb,
		b8g8r8a8_unorm,
		r16g16b16a16_sfloat,
		r32_float,
		r32g32b32a32_sfloat,
		d24_unorm_s8_uint,
	};

	struct Pipeline { u64 id; };

	enum struct DepthTest {
		write_on_test_off,
		write_on_test_on,
		write_off_test_on,
		write_off_test_off,
	};

	enum struct CullMode {
		none,
		front,
		back,
	};

	enum struct BlendMode {
		none,
		alpha_blend,
	};

	struct PipelineGraphicsCreateInfo {
		char const * pathMesh;
		char const * pathFragment;
		srat::slice<ImageFormat const> attachmentColorFormats;
		ImageFormat attachmentDepthStencilFormat;
		DepthTest depthTest;
		CullMode cullMode;
		BlendMode blendMode;
		srat::slice<char const * const> defines {};
		srat::slice<char const * const> includePaths {};
	};

	struct PipelineComputeCreateInfo {
		char const * pathCompute;
		srat::slice<char const * const> defines {};
		srat::slice<char const * const> includePaths {};
	};

	Pipeline pipeline_graphics_create(
		PipelineGraphicsCreateInfo const & createInfo
	);

	Pipeline pipeline_compute_create(
		PipelineComputeCreateInfo const & createInfo
	);

	void pipeline_destroy(Pipeline const & pipeline);

}

// -----------------------------------------------------------------------------
// -- acceleration structure
// -----------------------------------------------------------------------------

namespace vkof
{
	struct AccelerationStructureBlas { u64 id; };
	struct AccelerationStructureTlas { u64 id; };

	struct BlasCreateInfo {
		u64 positionVa;
		u32 vertexCount;
		u64 indexVa;
		u32 triangleCount;

		bool isOpaque { true };
	};

	AccelerationStructureBlas blas_create(BlasCreateInfo const & createInfo);
	void blas_destroy(AccelerationStructureBlas const & accelerationStructure);

	struct TlasInstance {
		AccelerationStructureBlas blas;
		f32m44 transform;
		u32 instanceCustomIndex;
		u32 rayMask;
	};

	struct TlasCreateInfo {
		u32 maxInstances;
	};

	AccelerationStructureTlas tlas_create(TlasCreateInfo const & createInfo);
	void tlas_destroy(AccelerationStructureTlas const & accelerationStructure);

	[[nodiscard]] u64 acceleration_structure_device_address(
		AccelerationStructureTlas const & as
	);

	// writes TLAS into slot at set=0 binding=3 (vkofTlas in GLSL)
	// All subsequent ray queries in any shader will search through this TLAS
	void acceleration_structure_set_tlas(AccelerationStructureTlas const & as);
}

// -----------------------------------------------------------------------------
// -- command buffer
// -----------------------------------------------------------------------------

namespace vkof
{
	struct CommandBuffer { u64 id; };
}

// -----------------------------------------------------------------------------
// -- buffer
// -----------------------------------------------------------------------------

namespace vkof
{
	struct Buffer { u64 id; };

	enum struct BufferMemory : u32 {
		DeviceOnly,
		HostWritable,
	};

	struct BufferCreateInfo {
		u64 byteCount;
		BufferMemory memory;
	};

	Buffer buffer_create(
		BufferCreateInfo const & createInfo,
		std::source_location loc = std::source_location::current()
	);
	void buffer_destroy(Buffer const & buffer);

	u64 buffer_virtual_address(Buffer const & buffer);
	srat::slice<u8> buffer_host_address(Buffer const & buffer);

	struct BufferUploadInfo {
		Buffer buffer;
		u64 byteOffset;
		srat::slice<u8 const> data;
	};
	void buffer_upload(BufferUploadInfo const & uploadInfo);
}

// -----------------------------------------------------------------------------
// -- image
// -----------------------------------------------------------------------------

namespace vkof
{
	struct Image { u64 id; };

	struct ImageCreateInfo {
		u32 width;
		u32 height;
		u32 depth { 1u };
		ImageFormat format;
		u32 mipLevels;
		srat::slice<u8 const> optInitialData;
	};

	Image image_create(
		ImageCreateInfo const & createInfo,
		std::source_location loc = std::source_location::current()
	);
	void image_destroy(Image const & image);

	u32 image_width(Image const & image);
	u32 image_height(Image const & image);
	u32 image_depth(Image const & image);
	void image_generate_mipmaps(Image const & image);

	struct Sampler { u64 id; };

	enum struct SamplerFilter {
		nearest,
		linear,
	};

	enum struct SamplerAddressMode {
		repeat,
		mirrored_repeat,
		clamp_to_edge,
	};

	enum struct SamplerMipmapMode {
		nearest,
		linear,
	};

	struct SamplerCreateInfo {
		SamplerFilter magFilter;
		SamplerFilter minFilter;
		SamplerAddressMode addressModeU;
		SamplerAddressMode addressModeV;
		SamplerAddressMode addressModeW;
		SamplerMipmapMode mipmapMode { SamplerMipmapMode::linear };
		f32 maxAnisotropy { 0.0f };
	};
	Sampler sampler_create(SamplerCreateInfo const & createInfo);
	void sampler_destroy(Sampler const & sampler);

	// for use with sampler2d (binding 0)
	struct ImageSamplerHandleInfo {
		Image image;
		Sampler sampler;
	};
	u32 image_sampler_handle(ImageSamplerHandleInfo const & info);
	[[nodiscard]] ImTextureID image_imgui_id(ImageSamplerHandleInfo const & info);
	void image_imgui_id_destroy(ImTextureID id);

	// for use with image2d (bindings 1/2)
	struct ImageStorageHandleInfo {
		Image image;
		u32 mipLevel;
	};
	u32 image_storage_handle(ImageStorageHandleInfo const & info);

	// for use with sampler3d (binding 4)
	u32 image_sampler3d_handle(ImageSamplerHandleInfo const & info);

	// for use with image3d (binding 5)
	u32 image_storage3d_handle(ImageStorageHandleInfo const & info);
}

// -----------------------------------------------------------------------------
// -- draw/dispatch
// -----------------------------------------------------------------------------

namespace vkof
{
	struct CmdDraw {
		CommandBuffer cmd;
		Pipeline pipeline;
		srat::slice<u8 const> pushconstant;
		u32 vertexCount;
		u32 instanceCount;
	};
	void cmd_draw(CmdDraw const & draw);

	struct CmdDispatch {
		CommandBuffer cmd;
		Pipeline pipeline;
		srat::slice<u8 const> pushconstant;
		u32v3 threadgroupSize;
		u32v3 invocationCount;
	};
	void cmd_dispatch(CmdDispatch const & dispatch);

	template<typename PushT>
	struct CmdDispatchPushconst {
		CommandBuffer cmd;
		Pipeline pipeline;
		PushT push;
		u32v3 threadgroupSize;
		u32v3 invocationCount;
	};
	template<typename PushT>
	inline void cmd_dispatch_pushconst(CmdDispatchPushconst<PushT> const & info)
	{
		cmd_dispatch({
			.cmd = info.cmd,
			.pipeline = info.pipeline,
			.pushconstant = srat::slice_as_bytes(info.push),
			.threadgroupSize = info.threadgroupSize,
			.invocationCount = info.invocationCount,
		});
	}

	template<typename PushT>
	struct CmdDrawPushconst {
		CommandBuffer cmd;
		Pipeline pipeline;
		PushT push;
		u32 vertexCount;
		u32 instanceCount;
	};
	template<typename PushT>
	inline void cmd_draw_pushconst(CmdDrawPushconst<PushT> const & info)
	{
		cmd_draw({
			.cmd = info.cmd,
			.pipeline = info.pipeline,
			.pushconstant = srat::slice_as_bytes(info.push),
			.vertexCount = info.vertexCount,
			.instanceCount = info.instanceCount,
		});
	}

	struct CmdCopyImage {
		CommandBuffer cmd;
		Image srcImage;
		u32 srcBaseMipLevel { 0 };
		u32 srcOffsetX { 0 };
		u32 srcOffsetY { 0 };
		Image dstImage;
		u32 dstBaseMipLevel { 0 };
		u32 dstOffsetX { 0 };
		u32 dstOffsetY { 0 };
		u32 width;
		u32 height;
	};
	void cmd_copy_image(CmdCopyImage const & copy);

	void tlas_build(
		CommandBuffer const & cmd,
		AccelerationStructureTlas const & tlas,
		srat::slice<TlasInstance const> const & instances
	);
}

// -----------------------------------------------------------------------------
// -- renderpass
// -----------------------------------------------------------------------------

namespace vkof
{
	struct RenderNode { u64 id; };
	struct TransientImage { u64 id; };
	struct TransientBuffer { u64 id; };

	enum struct RenderNodeAccess { read, write, readWrite, };
	enum struct RenderNodeLoadOp { load, clear, discard, };

	enum struct CommandQueue {
		graphics,
		compute,
	};

	struct TransientImageCreateInfo {
		ImageFormat format;
		f32 scaleWidth;
		f32 scaleHeight;
		u32 mipLevels;
		bool isDoubleBuffered;
	};

	TransientImage transient_image_create(
		TransientImageCreateInfo const & createInfo
	);

	struct TransientBufferCreateInfo {
		u64 byteCount;
		bool isDoubleBuffered;
	};
	TransientBuffer transient_buffer_create(
		TransientBufferCreateInfo const & createInfo
	);

	Image transient_image_get_image(TransientImage const & transientImage);
	Buffer transient_buffer_get_buffer(TransientBuffer const & transientBuffer);

	// Set the internal render resolution. Transient images are sized from
	//   this extent (times their scale factors), decoupled from the
	//   swapchain. Waits for the device to go idle and recreates the backing
	//   images of every live transient image; TransientImage handles held by
	//   the caller stay valid. Defaults to the swapchain extent at init.
	void render_extent_set(u32 const width, u32 const height);

	struct TransientImageStorageHandleInfo {
		TransientImage image;
		u32 mipLevel;
	};
	u32 transient_image_storage_handle(
		TransientImageStorageHandleInfo const & info
	);

	struct RenderNodeCreateInfo {
		CommandQueue queue;
	};
	RenderNode render_node_create(RenderNodeCreateInfo const & createInfo);
	void render_node_destroy(RenderNode const & node);

	struct RenderNodeImageInfo {
		RenderNode node;
		TransientImage image;
		RenderNodeAccess access;
	};
	void render_node_add_image(RenderNodeImageInfo const & info);

	struct RenderNodePersistentImageInfo {
		RenderNode node;
		Image image;
		RenderNodeAccess access;
	};
	void render_node_add_persistent_image(RenderNodePersistentImageInfo const & info);

	struct RenderNodeBufferInfo {
		RenderNode node;
		TransientBuffer buffer;
		RenderNodeAccess access;
	};
	void render_node_add_buffer(RenderNodeBufferInfo const & info);

	struct RenderNodeAttachmentColorInfo {
		RenderNode node;
		TransientImage image;
		RenderNodeLoadOp loadOp;
		u32 mipLevel;
		u32 colorIndex;
		srat::slice<f32 const> clearColor;
	};
	void render_node_attachment_color(
		RenderNodeAttachmentColorInfo const & info
	);

	struct RenderNodeAttachmentDepthInfo {
		RenderNode node;
		TransientImage image;
		RenderNodeLoadOp loadOp;
		u32 mipLevel;
		srat::slice<f32 const> clearDepth;
	};
	void render_node_attachment_depth(
		RenderNodeAttachmentDepthInfo const & info
	);

	struct RenderNodeCallbackInfo {
		RenderNode node;
		std::function<void(vkof::CommandBuffer const & cmd)> callback;
	};
	void render_node_callback(RenderNodeCallbackInfo const & info);

	struct RenderGraphExecuteInfo {
		srat::slice<RenderNode const> nodes;
		srat::slice<u8 const> rootPushconstant;
		// if set, vkof copies this image to the swapchain before ImGui
		TransientImage finalImage { 0 };
		f32m44 debugDrawViewProj {};
		// if set, debug lines depth-test against this image
		TransientImage debugDrawDepth { 0 };
	};
	void render_graph_execute(RenderGraphExecuteInfo const & executeInfo);
}

// -----------------------------------------------------------------------------
// -- lifetime
// -----------------------------------------------------------------------------

namespace vkof
{
	void init();
	// Headless init: no window, no surface, no swapchain.
	// Uses a fixed offscreen extent (default 512x512).
	// render_graph_execute skips acquire/present in this mode.
	void init_headless(u32 width = 512, u32 height = 512);
	void shutdown();
	void device_wait_idle();
	GLFWwindow * window();

	// Call once per frame before building any ImGui widgets.
	// render_graph_execute renders and presents the UI automatically.
	void imgui_begin();

	// Save a transient image to a PNG file. Blocks until the GPU is idle.
	void screenshot(TransientImage const & image, char const * const path);

	// returns the debug printf message captured this frame, or nullptr if none.
	// cleared at the start of each render_graph_execute. safe to call multiple times.
	[[nodiscard]] char const * probe_message();

	// returns true if any pipeline was successfully hot-reloaded this frame.
	// cleared at the start of each render_graph_execute.
	[[nodiscard]] bool shader_reloaded();
}

// -----------------------------------------------------------------------------
// -- debug draw
// -----------------------------------------------------------------------------

namespace vkof
{
	// deferred line-list draw; flushed in render_graph_execute before ImGui.
	// cleared automatically each frame.
	void debug_draw_line(f32v3 const & a, f32v3 const & b, f32v3 const & color);
	void debug_draw_box(f32v3 const & min, f32v3 const & max, f32v3 const & color);
	void debug_draw_sphere(f32v3 const & center, f32 radius, f32v3 const & color);

	// deferred solid-sphere draw with a user-supplied pipeline.
	// cleared automatically each frame.
	// requires exec.debugDrawDepth to be set.
	//
	// frag shader interface:
	//   layout(location = 0) in vec3 inNormal;
	//   layout(location = 1) perprimitiveEXT flat in uint inSphereIdx;
	struct DebugSphere {
		f32v3 position;
		f32 radius;
	};
	void debug_draw_spheres(
		srat::slice<DebugSphere const> const & spheres,
		Pipeline const pipeline
	);

	struct DebugSpherePipelineCreateInfo {
		char const * pathFrag;
		srat::slice<char const * const> includePaths {};
		srat::slice<char const * const> defines {};
	};
	// creates a pipeline using vkof's sphere mesh shader + the given frag shader.
	// hot-reload is supported on the frag shader path.
	Pipeline debug_sphere_pipeline_create(
		DebugSpherePipelineCreateInfo const & createInfo
	);
}

// -----------------------------------------------------------------------------
// -- buffer (extended)
// -----------------------------------------------------------------------------

namespace vkof
{
	// GPU -> CPU blocking copy. Symmetric counterpart to buffer_upload.
	struct BufferDownloadInfo {
		Buffer buffer;
		u64 byteOffset;
		srat::slice<u8> dst;  // caller-owned destination slice
	};
	void buffer_download(BufferDownloadInfo const & info);
}
