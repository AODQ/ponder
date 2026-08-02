#pragma once

// vkof internals shared with the ngx integration; only built with PONDER_DLSS

#include <vkof/vkof.hpp>

#include <vulkan/vulkan.h>

#include <vector>

namespace vkof::internal {
	struct NativeImage {
		VkImage image;
		VkImageView view;
		VkFormat format;
		u32 width;
		u32 height;
	};

	VkInstance native_instance();
	VkPhysicalDevice native_physical_device();
	VkDevice native_device();
	VkCommandBuffer native_command_buffer(CommandBuffer const & cmd);
	[[nodiscard]] bool native_image(Image const & image, NativeImage & out);

	// commands recorded outside vkof clobber its bound pipeline, descriptor
	// set and root push constants; the next dispatch rebinds them
	void command_buffer_state_invalidate(CommandBuffer const & cmd);
}

namespace vkof::ngx {
	void required_extensions(
		std::vector<char const *> & instanceExtensions,
		std::vector<char const *> & deviceExtensions
	);
	void init();
	void shutdown();
}
