#pragma once

#include "vk_context.h"
#include "scene.h"
#include "randomizer.h"

#include <glm/glm.hpp>
#include <vector>

namespace dart {

struct AccelBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    VkDeviceAddress deviceAddress = 0;
};

struct BLAS {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    AccelBuffer buffer;
};

// Per-frame TLAS resources (double-buffered to avoid races between frames)
struct TLASFrame {
    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
    AccelBuffer tlasBuffer;
    AccelBuffer instanceBuffer;
    /// One GpuMaterial per instance, indexed by instanceCustomIndex. Written
    /// alongside the instance buffer so the two never disagree about what an
    /// index means.
    AccelBuffer materialBuffer;
    AccelBuffer scratchBuffer;
    VkDescriptorSet tlasDescSet = VK_NULL_HANDLE;
};

struct AccelStructure {
    static constexpr int FRAMES_IN_FLIGHT = 2;

    std::vector<BLAS> blasList;  // one per mesh in scene.allMeshes_ (shared)
    TLASFrame tlasFrames[FRAMES_IN_FLIGHT];  // per-frame TLAS resources

    uint32_t maxInstances = 0;

    void init(VkContext& ctx, Scene& scene, VkDescriptorPool pool,
              VkDescriptorSetLayout tlasLayout);
    void updateTLAS(VkContext& ctx, VkCommandBuffer cmd, Scene& scene,
                    const FrameState& state, int frameIndex);

    /// Rebuild the structures for a handful of meshes whose buffers have been
    /// replaced -- the rolling dart refresh. The caller must have waited for
    /// the device first; the old buffers are gone by the time this runs.
    void refreshBLAS(VkContext& ctx, Scene& scene,
                     const std::vector<size_t>& meshIndices);
    VkDescriptorSet currentDescSet(int frameIndex) const {
        return tlasFrames[frameIndex].tlasDescSet;
    }
    void destroy(VkContext& ctx);

private:
    void buildBLAS(VkContext& ctx, Scene& scene);
    void buildBLASFor(VkContext& ctx, Scene& scene,
                      const std::vector<size_t>& indices);
    void createTLAS(VkContext& ctx, Scene& scene, VkDescriptorPool pool,
                    VkDescriptorSetLayout tlasLayout);
    AccelBuffer createBuffer(VkContext& ctx, VkDeviceSize size,
                             VkBufferUsageFlags usage, bool hostVisible);
};

/// Roll one dart design out of the pool, if this frame is due one.
///
/// Waits for the device first: the refresh frees the buffers the retired
/// design was drawn from, and with two frames in flight one of them may still
/// be reading them. At one refresh every kDartRefreshFrames the stall is under
/// a percent of throughput, and it removes a whole class of use-after-free
/// that a "skip the variants in flight" scheme would only make unlikely.
///
/// `inFlightVariant` is the design the frame about to be drawn uses; it is
/// never the one retired, so a refresh cannot change the geometry out from
/// under the annotation that was just computed for it.
void rollDartVariant(VkContext& ctx, Scene& scene, AccelStructure& accel,
                     int frameId, int inFlightVariant);

} // namespace dart
