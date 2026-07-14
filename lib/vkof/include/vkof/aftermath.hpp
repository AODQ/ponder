#pragma once
#ifdef VKOF_AFTERMATH

#include <cstdint>
#include <vulkan/vulkan.h>

void vkof_aftermath_enable();
void vkof_aftermath_disable();
void vkof_aftermath_on_device_lost();
void vkof_aftermath_set_handles(
	VkDevice device,
	VkQueue graphics_queue,
	PFN_vkGetQueueCheckpointDataNV pfn_get_checkpoints
);
void vkof_aftermath_register_spirv(void const * pData, uint32_t size);

#endif
