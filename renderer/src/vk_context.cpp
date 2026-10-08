#include "vk_context.h"

#include <VkBootstrap.h>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define VK_CHECK(x)                                                     \
    do {                                                                \
        VkResult _r = (x);                                             \
        if (_r != VK_SUCCESS) {                                        \
            char buf[256];                                             \
            snprintf(buf, sizeof(buf),                                 \
                     "Vulkan error %d at %s:%d", _r, __FILE__, __LINE__); \
            throw std::runtime_error(buf);                             \
        }                                                               \
    } while (0)

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

namespace dart {

// RT extensions we require
static const char* RT_EXTENSIONS[] = {
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
};
static constexpr uint32_t RT_EXTENSION_COUNT = sizeof(RT_EXTENSIONS) / sizeof(RT_EXTENSIONS[0]);

// Check if a physical device supports all RT extensions
static bool deviceSupportsRT(VkPhysicalDevice pd) {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, exts.data());

    for (uint32_t i = 0; i < RT_EXTENSION_COUNT; ++i) {
        bool found = false;
        for (auto& ext : exts) {
            if (strcmp(ext.extensionName, RT_EXTENSIONS[i]) == 0) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

// The raster fallback (surface_raster.frag) approximates what surface_rt.frag
// traces -- shadows, ambient occlusion, reflections -- and its frames are
// measurably brighter and more evenly lit. Fine for looking at geometry and
// labels; not the distribution the models are trained on.
static void logRtStatus(bool rtEnabled) {
    if (rtEnabled) {
        fprintf(stderr, "VkContext: ray tracing ENABLED\n");
    } else {
        fprintf(stderr,
                "VkContext: WARNING: ray tracing not available. Rendering with "
                "the raster fallback, whose lighting does not match the "
                "ray-traced path; do not train on this output.\n");
    }
}

// Core features the renderer depends on, checked against `pd` and returned
// ready to pass as VkDeviceCreateInfo::pEnabledFeatures.
//
// samplerAnisotropy is required rather than optional: the texture sampler
// asks for it, and without it the decal atlas and the background probe are
// sampled isotropically, which blurs printed marks on a board seen at grazing
// angles -- a silent change to the training distribution, not a crash.
static VkPhysicalDeviceFeatures requiredDeviceFeatures(VkPhysicalDevice pd) {
    VkPhysicalDeviceFeatures supported = {};
    vkGetPhysicalDeviceFeatures(pd, &supported);
    if (!supported.samplerAnisotropy) {
        throw std::runtime_error(
            "Selected Vulkan device does not support samplerAnisotropy");
    }
    VkPhysicalDeviceFeatures enabled = {};
    enabled.samplerAnisotropy = VK_TRUE;
    return enabled;
}

static const char* deviceTypeName(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "CPU/software";
        default:                                     return "other";
    }
}

// Resolve which physical device to use, logging the full enumeration.
//
// Enumeration order is not stable across reboots, so a raw index is a poor
// way to name a device — the same index can be a discrete GPU one day and a
// software rasteriser the next, which shows up as the renderer running at a
// few fps instead of several hundred.  When no index is forced we therefore
// choose by device *type*, never picking a CPU/software device, and allow a
// name substring override via DARTS_GPU_NAME (e.g. "NVIDIA").
//
// A UUID, when given, wins over everything: it names one physical GPU
// whatever the enumeration order, and NVIDIA's Vulkan deviceUUID is the same
// UUID CUDA reports, so a process can render on exactly the GPU it trains on
// (or on a chosen other one) without mapping between the two numberings.
static std::string uuidString(const uint8_t* u) {
    char buf[37];
    snprintf(buf, sizeof(buf),
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10],
             u[11], u[12], u[13], u[14], u[15]);
    return buf;
}

static std::string normaliseUuid(std::string s) {
    if (s.rfind("GPU-", 0) == 0) s = s.substr(4);
    std::string out;
    for (char c : s) {
        if (c == '-') continue;
        out += (char)std::tolower((unsigned char)c);
    }
    return out;
}

static int resolveGpuIndex(VkInstance instance, int requested,
                           const std::string& uuid) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) {
        throw std::runtime_error("No Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    std::vector<VkPhysicalDeviceProperties> props(count);
    std::vector<std::string> uuids(count);
    fprintf(stderr, "VkContext: %u Vulkan device(s):\n", count);
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceIDProperties idProps = {};
        idProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        VkPhysicalDeviceProperties2 props2 = {};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &idProps;
        vkGetPhysicalDeviceProperties2(devices[i], &props2);
        props[i] = props2.properties;
        uuids[i] = uuidString(idProps.deviceUUID);
        fprintf(stderr, "  [%u] %s (%s) uuid %s\n", i, props[i].deviceName,
                deviceTypeName(props[i].deviceType), uuids[i].c_str());
    }

    if (!uuid.empty()) {
        const std::string want = normaliseUuid(uuid);
        for (uint32_t i = 0; i < count; ++i) {
            if (normaliseUuid(uuids[i]) == want) return (int)i;
        }
        throw std::runtime_error("No Vulkan device with UUID " + uuid);
    }

    if (requested >= 0) {
        if (requested >= (int)count) {
            throw std::runtime_error(
                "GPU index " + std::to_string(requested) +
                " out of range (have " + std::to_string(count) + " devices)");
        }
        return requested;
    }

    if (const char* want = getenv("DARTS_GPU_NAME")) {
        for (uint32_t i = 0; i < count; ++i) {
            if (strstr(props[i].deviceName, want) != nullptr) {
                fprintf(stderr, "VkContext: matched DARTS_GPU_NAME=%s\n", want);
                return (int)i;
            }
        }
        fprintf(stderr, "VkContext: DARTS_GPU_NAME=%s matched nothing, "
                        "falling back to type preference\n", want);
    }

    for (auto type : {VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                      VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
                      VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU}) {
        for (uint32_t i = 0; i < count; ++i) {
            if (props[i].deviceType == type) return (int)i;
        }
    }

    fprintf(stderr, "VkContext: WARNING no GPU found, only software rendering "
                    "is available — expect a few fps rather than hundreds\n");
    return 0;
}

void VkContext::init(int gpuIndex, const std::string& gpuUuid) {
    // Instance — Vulkan 1.2 (bufferDeviceAddress is core in 1.2)
    // vk-bootstrap automatically enables VK_KHR_portability_enumeration on macOS
    vkb::InstanceBuilder instBuilder;
    auto instResult = instBuilder
        .set_app_name("dartboard_gen")
        .require_api_version(1, 2, 0)
        .set_headless()
        .build();
    if (!instResult) {
        throw std::runtime_error(
            std::string("Failed to create Vulkan instance: ") +
            instResult.error().message());
    }
    auto vkbInst = instResult.value();
    instance = vkbInst.instance;

    // Feature structs for RT (chained via pNext when rtEnabled)
    VkPhysicalDeviceVulkan12Features features12 = {};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeatures = {};
    accelFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accelFeatures.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = {};
    rayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rayQueryFeatures.rayQuery = VK_TRUE;

    // Physical device.  Both the explicit-index path (for DDP multi-GPU) and
    // the automatic path resolve to a concrete index here, so the selection is
    // deterministic and logged rather than delegated to vk-bootstrap, whose
    // prefer_gpu_device_type is only a preference and will silently fall back
    // to a software device.
    gpuIndex = resolveGpuIndex(instance, gpuIndex, gpuUuid);
    if (gpuIndex >= 0) {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        physicalDevice = devices[gpuIndex];
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        fprintf(stderr, "VkContext: selected GPU %d: %s\n", gpuIndex, props.deviceName);

        // Check RT support
        rtEnabled = deviceSupportsRT(physicalDevice);
        logRtStatus(rtEnabled);

        // Find a graphics queue family
        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfProps(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, qfProps.data());
        queueFamily = UINT32_MAX;
        for (uint32_t i = 0; i < qfCount; ++i) {
            if (qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                queueFamily = i;
                break;
            }
        }
        if (queueFamily == UINT32_MAX) {
            throw std::runtime_error(
                "No graphics queue family on GPU " + std::to_string(gpuIndex));
        }

        // Create logical device
        float priority = 1.0f;
        VkDeviceQueueCreateInfo queueCI = {};
        queueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCI.queueFamilyIndex = queueFamily;
        queueCI.queueCount = 1;
        queueCI.pQueuePriorities = &priority;

        // Collect enabled extensions
        std::vector<const char*> enabledExts;
        if (rtEnabled) {
            for (uint32_t i = 0; i < RT_EXTENSION_COUNT; ++i)
                enabledExts.push_back(RT_EXTENSIONS[i]);
        }

        const VkPhysicalDeviceFeatures coreFeatures =
            requiredDeviceFeatures(physicalDevice);

        VkDeviceCreateInfo devCI = {};
        devCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        devCI.queueCreateInfoCount = 1;
        devCI.pQueueCreateInfos = &queueCI;
        devCI.enabledExtensionCount = (uint32_t)enabledExts.size();
        devCI.ppEnabledExtensionNames = enabledExts.data();
        devCI.pEnabledFeatures = &coreFeatures;

        // Chain feature structs when RT is available
        if (rtEnabled) {
            features12.pNext = &accelFeatures;
            accelFeatures.pNext = &rayQueryFeatures;
            devCI.pNext = &features12;
        }

        VK_CHECK(vkCreateDevice(physicalDevice, &devCI, nullptr, &device));
        vkGetDeviceQueue(device, queueFamily, 0, &graphicsQueue);
    } else {
        // vk-bootstrap path: try with RT extensions first, fall back without.
        // Features set on the selector are the ones DeviceBuilder enables.
        VkPhysicalDeviceFeatures coreFeatures = {};
        coreFeatures.samplerAnisotropy = VK_TRUE;

        vkb::PhysicalDeviceSelector selector(vkbInst);
        selector
            .set_minimum_version(1, 2)
            .defer_surface_initialization()
            .require_present(false)
            .set_required_features(coreFeatures)
            .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete);

        // Note: desired extensions are checked after select() via deviceSupportsRT()

        auto physResult = selector.select();
        if (!physResult) {
            throw std::runtime_error(
                std::string("Failed to select physical device: ") +
                physResult.error().message());
        }
        auto vkbPhys = physResult.value();
        physicalDevice = vkbPhys.physical_device;

        // Check if the selected device actually supports RT
        rtEnabled = deviceSupportsRT(physicalDevice);

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        fprintf(stderr, "VkContext: selected GPU: %s\n", props.deviceName);
        logRtStatus(rtEnabled);

        // If RT supported, re-select with required extensions
        if (rtEnabled) {
            vkb::PhysicalDeviceSelector rtSelector(vkbInst);
            rtSelector
                .set_minimum_version(1, 2)
                .defer_surface_initialization()
                .require_present(false)
                .set_required_features(coreFeatures)
                .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete);
            for (uint32_t i = 0; i < RT_EXTENSION_COUNT; ++i)
                rtSelector.add_required_extension(RT_EXTENSIONS[i]);

            auto rtPhysResult = rtSelector.select();
            if (rtPhysResult) {
                // The RT selector may settle on a different device than the
                // first pass did; everything downstream (VMA, limits, the
                // logical device) must use the one the device is built on.
                vkbPhys = rtPhysResult.value();
                physicalDevice = vkbPhys.physical_device;
            } else {
                rtEnabled = false;
                logRtStatus(rtEnabled);
            }
        }

        vkb::DeviceBuilder devBuilder(vkbPhys);

        // Chain feature structs when RT is available
        if (rtEnabled) {
            features12.pNext = &accelFeatures;
            accelFeatures.pNext = &rayQueryFeatures;
            devBuilder.add_pNext(&features12);
        }

        auto devResult = devBuilder.build();
        if (!devResult) {
            throw std::runtime_error(
                std::string("Failed to create logical device: ") +
                devResult.error().message());
        }
        auto vkbDev = devResult.value();
        device = vkbDev.device;

        auto queueResult = vkbDev.get_queue(vkb::QueueType::graphics);
        if (!queueResult) {
            throw std::runtime_error("Failed to get graphics queue");
        }
        graphicsQueue = queueResult.value();
        queueFamily = vkbDev.get_queue_index(vkb::QueueType::graphics).value();
    }

    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        maxSamplerAnisotropy = props.limits.maxSamplerAnisotropy;
    }

    // VMA allocator
    VmaAllocatorCreateInfo allocInfo = {};
    allocInfo.physicalDevice = physicalDevice;
    allocInfo.device = device;
    allocInfo.instance = instance;
    allocInfo.vulkanApiVersion = VK_API_VERSION_1_2;
    if (rtEnabled) {
        allocInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }
    VK_CHECK(vmaCreateAllocator(&allocInfo, &allocator));

    // Command pool
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool));

    // Load RT function pointers
    if (rtEnabled) {
        auto load = [&](const char* name) {
            return vkGetDeviceProcAddr(device, name);
        };
        vkCreateAccelerationStructureKHR =
            (PFN_vkCreateAccelerationStructureKHR)load("vkCreateAccelerationStructureKHR");
        vkDestroyAccelerationStructureKHR =
            (PFN_vkDestroyAccelerationStructureKHR)load("vkDestroyAccelerationStructureKHR");
        vkGetAccelerationStructureBuildSizesKHR =
            (PFN_vkGetAccelerationStructureBuildSizesKHR)load("vkGetAccelerationStructureBuildSizesKHR");
        vkCmdBuildAccelerationStructuresKHR =
            (PFN_vkCmdBuildAccelerationStructuresKHR)load("vkCmdBuildAccelerationStructuresKHR");
        vkGetAccelerationStructureDeviceAddressKHR =
            (PFN_vkGetAccelerationStructureDeviceAddressKHR)load("vkGetAccelerationStructureDeviceAddressKHR");
        // bufferDeviceAddress is core in Vulkan 1.2 — try core name first, then KHR
        vkGetBufferDeviceAddressKHR =
            (PFN_vkGetBufferDeviceAddressKHR)load("vkGetBufferDeviceAddress");
        if (!vkGetBufferDeviceAddressKHR)
            vkGetBufferDeviceAddressKHR =
                (PFN_vkGetBufferDeviceAddressKHR)load("vkGetBufferDeviceAddressKHR");

        // Validate all function pointers loaded
        if (!vkCreateAccelerationStructureKHR || !vkDestroyAccelerationStructureKHR ||
            !vkGetAccelerationStructureBuildSizesKHR || !vkCmdBuildAccelerationStructuresKHR ||
            !vkGetAccelerationStructureDeviceAddressKHR || !vkGetBufferDeviceAddressKHR) {
            fprintf(stderr, "ERROR: Failed to load RT function pointers:\n");
            fprintf(stderr, "  vkCreateAccelerationStructureKHR: %p\n", (void*)vkCreateAccelerationStructureKHR);
            fprintf(stderr, "  vkDestroyAccelerationStructureKHR: %p\n", (void*)vkDestroyAccelerationStructureKHR);
            fprintf(stderr, "  vkGetAccelerationStructureBuildSizesKHR: %p\n", (void*)vkGetAccelerationStructureBuildSizesKHR);
            fprintf(stderr, "  vkCmdBuildAccelerationStructuresKHR: %p\n", (void*)vkCmdBuildAccelerationStructuresKHR);
            fprintf(stderr, "  vkGetAccelerationStructureDeviceAddressKHR: %p\n", (void*)vkGetAccelerationStructureDeviceAddressKHR);
            fprintf(stderr, "  vkGetBufferDeviceAddressKHR: %p\n", (void*)vkGetBufferDeviceAddressKHR);
            fflush(stderr);
            rtEnabled = false;
            logRtStatus(rtEnabled);
        }
    }
}

void VkContext::destroy() {
    if (device) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        vmaDestroyAllocator(allocator);
        vkDestroyDevice(device, nullptr);
    }
    if (instance) {
        vkDestroyInstance(instance, nullptr);
    }
}

VkCommandBuffer VkContext::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &cmd));

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    return cmd;
}

void VkContext::endSingleTimeCommands(VkCommandBuffer cmd) {
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(graphicsQueue));

    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

} // namespace dart
