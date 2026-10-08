#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vk_mem_alloc.h>

namespace dart {

struct VkContext {
    VkInstance       instance       = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice         device         = VK_NULL_HANDLE;
    VkQueue          graphicsQueue  = VK_NULL_HANDLE;
    uint32_t         queueFamily    = 0;
    VmaAllocator     allocator      = VK_NULL_HANDLE;
    VkCommandPool    commandPool    = VK_NULL_HANDLE;

    // Ray tracing support (VK_KHR_ray_query)
    bool rtEnabled = false;

    /// VkPhysicalDeviceLimits::maxSamplerAnisotropy of the selected device.
    /// samplerAnisotropy is required and enabled at device creation.
    float maxSamplerAnisotropy = 1.0f;

    // RT function pointers (loaded dynamically when rtEnabled)
    PFN_vkCreateAccelerationStructureKHR           vkCreateAccelerationStructureKHR = nullptr;
    PFN_vkDestroyAccelerationStructureKHR          vkDestroyAccelerationStructureKHR = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR    vkGetAccelerationStructureBuildSizesKHR = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR        vkCmdBuildAccelerationStructuresKHR = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR = nullptr;
    PFN_vkGetBufferDeviceAddressKHR                vkGetBufferDeviceAddressKHR = nullptr;

    /// `gpuUuid`, if non-empty, selects the device by UUID (CUDA's form, with
    /// or without the "GPU-" prefix) and takes precedence over `gpuIndex`.
    void init(int gpuIndex = -1, const std::string& gpuUuid = "");
    void destroy();

    // Utility: one-shot command buffer
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer cmd);
};

} // namespace dart
