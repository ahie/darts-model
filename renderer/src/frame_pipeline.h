#pragma once

#include "vk_context.h"
#include "render_pass.h"

#include <functional>
#include <future>
#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace dart {

// Per-frame GPU resources (double-buffered)
struct FrameResources {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence         fence         = VK_NULL_HANDLE;

    // MSAA color image (multisampled render target)
    VkImage       msaaColorImage = VK_NULL_HANDLE;
    VmaAllocation msaaColorAlloc = VK_NULL_HANDLE;
    VkImageView   msaaColorView  = VK_NULL_HANDLE;

    // Resolve target for the MSAA pass, at the supersampled size.
    VkImage       colorImage     = VK_NULL_HANDLE;
    VmaAllocation colorAlloc     = VK_NULL_HANDLE;
    VkImageView   colorView      = VK_NULL_HANDLE;

    // Output-sized image the supersampled result is filtered down into, and
    // the one actually read back. Separate from colorImage because the render
    // targets are kSupersample times larger than the frame we return.
    VkImage       readbackImage  = VK_NULL_HANDLE;
    VmaAllocation readbackAlloc  = VK_NULL_HANDLE;

    // MSAA depth image
    VkImage       depthImage     = VK_NULL_HANDLE;
    VmaAllocation depthAlloc     = VK_NULL_HANDLE;
    VkImageView   depthView      = VK_NULL_HANDLE;

    // Framebuffer
    VkFramebuffer framebuffer    = VK_NULL_HANDLE;

    // Segmentation pass: class + instance ids at the supersampled size, single
    // sampled. Its own depth buffer because it renders separately -- ids
    // cannot go through the MSAA resolve or the supersample downsample, both
    // of which average.
    VkImage       segImage        = VK_NULL_HANDLE;
    VmaAllocation segAlloc        = VK_NULL_HANDLE;
    VkImageView   segView         = VK_NULL_HANDLE;
    VkImage       segDepthImage   = VK_NULL_HANDLE;
    VmaAllocation segDepthAlloc   = VK_NULL_HANDLE;
    VkImageView   segDepthView    = VK_NULL_HANDLE;
    VkFramebuffer segFramebuffer  = VK_NULL_HANDLE;
    VkBuffer      segStagingBuffer = VK_NULL_HANDLE;
    VmaAllocation segStagingAlloc  = VK_NULL_HANDLE;
    void*         mappedSegStaging = nullptr;
    // Depth from the same pass, so it is registered with the ids pixel for
    // pixel. Raw D32 window-space depth; converting it to a height above the
    // board plane needs the view matrix and near/far, which ride in the
    // annotation.
    VkBuffer      segDepthStagingBuffer = VK_NULL_HANDLE;
    VmaAllocation segDepthStagingAlloc  = VK_NULL_HANDLE;
    void*         mappedSegDepthStaging = nullptr;

    // Staging buffer for readback
    VkBuffer      stagingBuffer  = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc   = VK_NULL_HANDLE;
    void*         mappedStaging  = nullptr;

    // UBO
    VkBuffer      uboBuffer      = VK_NULL_HANDLE;
    VmaAllocation uboAlloc       = VK_NULL_HANDLE;
    void*         mappedUBO      = nullptr;
    VkDescriptorSet uboDescSet   = VK_NULL_HANDLE;
};

// Simple thread pool for JPEG encoding + file I/O
class ThreadPool {
public:
    explicit ThreadPool(int numThreads);
    ~ThreadPool();

    std::future<void> submit(std::function<void()> task);

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

struct FramePipeline {
    static constexpr int FRAMES_IN_FLIGHT = 2;

    FrameResources frames[FRAMES_IN_FLIGHT];
    int currentFrame = 0;
    /// Output size -- what waitAndReadback returns and what annotations are
    /// expressed in.
    uint32_t width = 0, height = 0;
    /// Render size, kSupersample times the output size.
    uint32_t renderWidth = 0, renderHeight = 0;

    /// JPEG encoding and file writes for dartboard_gen. The Python Renderer
    /// returns frames in memory and constructs this with zero threads.
    ThreadPool pool;
    std::vector<std::future<void>> pendingWrites;

    explicit FramePipeline(int poolThreads = 4);

    void init(VkContext& ctx, RenderPass& rp, uint32_t w, uint32_t h);
    void destroy(VkContext& ctx);

    // Get current frame resources (the one we're about to render into)
    FrameResources& current() { return frames[currentFrame]; }

    // Wait for this slot's previous submission (FRAMES_IN_FLIGHT frames ago)
    // and return its mapped RGBA pixels (width * height * 4 bytes).
    const void* waitAndReadback(VkContext& ctx);

    // After recording + submitting command buffer, advance to next frame
    void advance();

    // Drop futures whose task has finished, rethrowing any failure. Called
    // once per frame so pendingWrites stays bounded by what is in flight.
    void reapCompletedWrites();

    // Flush all pending write futures
    void flush();

    // Encode RGBA pixels to JPEG and write to file (dispatched to thread pool)
    void encodeAndWrite(const void* rgbaPixels, uint32_t w, uint32_t h,
                        const std::string& path, int quality = 90);

    // Write string to file (dispatched to thread pool)
    void writeFile(const std::string& path, const std::string& content);
};

} // namespace dart
