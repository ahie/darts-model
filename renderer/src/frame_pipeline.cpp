#include "frame_pipeline.h"

#include <turbojpeg.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

#define VK_CHECK(x)                                                     \
    do {                                                                \
        VkResult _r = (x);                                             \
        if (_r != VK_SUCCESS) {                                        \
            fprintf(stderr, "Vulkan error %d at %s:%d\n",              \
                    _r, __FILE__, __LINE__);                            \
            abort();                                                    \
        }                                                               \
    } while (0)

namespace dart {

// ---------------------------------------------------------------------------
// ThreadPool
// ---------------------------------------------------------------------------

ThreadPool::ThreadPool(int numThreads) {
    for (int i = 0; i < numThreads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock lock(mutex_);
                    cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty()) return;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
}

std::future<void> ThreadPool::submit(std::function<void()> task) {
    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future();
    {
        std::lock_guard lock(mutex_);
        tasks_.push([promise, task = std::move(task)]() mutable {
            try {
                task();
                promise->set_value();
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        });
    }
    cv_.notify_one();
    return future;
}

// ---------------------------------------------------------------------------
// FramePipeline
// ---------------------------------------------------------------------------

FramePipeline::FramePipeline(int poolThreads) : pool(poolThreads) {}

void FramePipeline::init(VkContext& ctx, RenderPass& rp, uint32_t w, uint32_t h) {
    // w/h are the OUTPUT size. Everything the render pass touches is built at
    // kSupersample times that and filtered down before readback, so `w`/`h`
    // below refer to the render targets while width/height stay the output.
    width = w;
    height = h;
    renderWidth = w * kSupersample;
    renderHeight = h * kSupersample;
    const uint32_t outW = w, outH = h;
    w = renderWidth;
    h = renderHeight;

    for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        auto& fr = frames[i];

        // Command buffer
        VkCommandBufferAllocateInfo cmdAllocInfo = {};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.commandPool = ctx.commandPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(ctx.device, &cmdAllocInfo, &fr.commandBuffer));

        // Fence (start signaled so first wait doesn't hang)
        VkFenceCreateInfo fenceCI = {};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(ctx.device, &fenceCI, nullptr, &fr.fence));

        // MSAA color image (multisampled render target)
        VkImageCreateInfo msaaColorCI = {};
        msaaColorCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        msaaColorCI.imageType = VK_IMAGE_TYPE_2D;
        msaaColorCI.format = kColorFormat;
        msaaColorCI.extent = {w, h, 1};
        msaaColorCI.mipLevels = 1;
        msaaColorCI.arrayLayers = 1;
        msaaColorCI.samples = rp.msaaSamples;
        msaaColorCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        msaaColorCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        msaaColorCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo msaaColorAllocCI = {};
        msaaColorAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        VK_CHECK(vmaCreateImage(ctx.allocator, &msaaColorCI, &msaaColorAllocCI,
                                &fr.msaaColorImage, &fr.msaaColorAlloc, nullptr));

        VkImageViewCreateInfo msaaColorViewCI = {};
        msaaColorViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        msaaColorViewCI.image = fr.msaaColorImage;
        msaaColorViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        msaaColorViewCI.format = kColorFormat;
        msaaColorViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        msaaColorViewCI.subresourceRange.levelCount = 1;
        msaaColorViewCI.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(ctx.device, &msaaColorViewCI, nullptr, &fr.msaaColorView));

        // Resolve color image (1 sample, supersampled size)
        VkImageCreateInfo colorCI = {};
        colorCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        colorCI.imageType = VK_IMAGE_TYPE_2D;
        colorCI.format = kColorFormat;
        colorCI.extent = {w, h, 1};
        colorCI.mipLevels = 1;
        colorCI.arrayLayers = 1;
        colorCI.samples = VK_SAMPLE_COUNT_1_BIT;
        colorCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        colorCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        colorCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo colorAllocCI = {};
        colorAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        VK_CHECK(vmaCreateImage(ctx.allocator, &colorCI, &colorAllocCI,
                                &fr.colorImage, &fr.colorAlloc, nullptr));

        VkImageViewCreateInfo colorViewCI = {};
        colorViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        colorViewCI.image = fr.colorImage;
        colorViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        colorViewCI.format = kColorFormat;
        colorViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        colorViewCI.subresourceRange.levelCount = 1;
        colorViewCI.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(ctx.device, &colorViewCI, nullptr, &fr.colorView));

        // Output-sized image the supersampled resolve is filtered into. Same
        // SRGB format as the resolve, so the blit decodes, filters in linear
        // and re-encodes, and the bytes copied back are sRGB.
        VkImageCreateInfo rbCI = colorCI;
        rbCI.extent = {outW, outH, 1};
        rbCI.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo rbAllocCI = {};
        rbAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        VK_CHECK(vmaCreateImage(ctx.allocator, &rbCI, &rbAllocCI,
                                &fr.readbackImage, &fr.readbackAlloc, nullptr));

        // MSAA depth image
        VkImageCreateInfo depthCI = {};
        depthCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        depthCI.imageType = VK_IMAGE_TYPE_2D;
        depthCI.format = VK_FORMAT_D32_SFLOAT;
        depthCI.extent = {w, h, 1};
        depthCI.mipLevels = 1;
        depthCI.arrayLayers = 1;
        depthCI.samples = rp.msaaSamples;
        depthCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        depthCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        depthCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo depthAllocCI = {};
        depthAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        VK_CHECK(vmaCreateImage(ctx.allocator, &depthCI, &depthAllocCI,
                                &fr.depthImage, &fr.depthAlloc, nullptr));

        VkImageViewCreateInfo depthViewCI = {};
        depthViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        depthViewCI.image = fr.depthImage;
        depthViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        depthViewCI.format = VK_FORMAT_D32_SFLOAT;
        depthViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        depthViewCI.subresourceRange.levelCount = 1;
        depthViewCI.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(ctx.device, &depthViewCI, nullptr, &fr.depthView));

        // Framebuffer: [msaaColor, depth, resolveColor]
        VkImageView attachments[] = {fr.msaaColorView, fr.depthView, fr.colorView};
        VkFramebufferCreateInfo fbCI = {};
        fbCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbCI.renderPass = rp.renderPass;
        fbCI.attachmentCount = 3;
        fbCI.pAttachments = attachments;
        fbCI.width = w;
        fbCI.height = h;
        fbCI.layers = 1;
        VK_CHECK(vkCreateFramebuffer(ctx.device, &fbCI, nullptr, &fr.framebuffer));

        // --- Segmentation attachments, at the supersampled size and 1 sample ---
        VkImageCreateInfo segCI = {};
        segCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        segCI.imageType = VK_IMAGE_TYPE_2D;
        segCI.format = VK_FORMAT_R8G8B8A8_UINT;
        segCI.extent = {w, h, 1};
        segCI.mipLevels = 1;
        segCI.arrayLayers = 1;
        segCI.samples = VK_SAMPLE_COUNT_1_BIT;
        segCI.tiling = VK_IMAGE_TILING_OPTIMAL;
        segCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        segCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo segAllocCI = {};
        segAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        VK_CHECK(vmaCreateImage(ctx.allocator, &segCI, &segAllocCI,
                                &fr.segImage, &fr.segAlloc, nullptr));

        VkImageViewCreateInfo segViewCI = {};
        segViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        segViewCI.image = fr.segImage;
        segViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
        segViewCI.format = VK_FORMAT_R8G8B8A8_UINT;
        segViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        segViewCI.subresourceRange.levelCount = 1;
        segViewCI.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(ctx.device, &segViewCI, nullptr, &fr.segView));

        VkImageCreateInfo segDepthCI = depthCI;
        segDepthCI.extent = {w, h, 1};
        segDepthCI.samples = VK_SAMPLE_COUNT_1_BIT;
        segDepthCI.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        VK_CHECK(vmaCreateImage(ctx.allocator, &segDepthCI, &depthAllocCI,
                                &fr.segDepthImage, &fr.segDepthAlloc, nullptr));

        VkImageViewCreateInfo segDepthViewCI = depthViewCI;
        segDepthViewCI.image = fr.segDepthImage;
        VK_CHECK(vkCreateImageView(ctx.device, &segDepthViewCI, nullptr, &fr.segDepthView));

        VkImageView segAttachments[] = {fr.segView, fr.segDepthView};
        VkFramebufferCreateInfo segFbCI = {};
        segFbCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        segFbCI.renderPass = rp.segRenderPass;
        segFbCI.attachmentCount = 2;
        segFbCI.pAttachments = segAttachments;
        segFbCI.width = w;
        segFbCI.height = h;
        segFbCI.layers = 1;
        VK_CHECK(vkCreateFramebuffer(ctx.device, &segFbCI, nullptr, &fr.segFramebuffer));

        // Staging buffer for readback (host-visible)
        // Readback is the output-sized image, not the supersampled one.
        VkDeviceSize stagingSize = outW * outH * 4;
        VkBufferCreateInfo stagingCI = {};
        stagingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingCI.size = stagingSize;
        stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VmaAllocationCreateInfo stagingAllocCI = {};
        stagingAllocCI.usage = VMA_MEMORY_USAGE_AUTO;
        stagingAllocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                               VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo stagingInfo;
        VK_CHECK(vmaCreateBuffer(ctx.allocator, &stagingCI, &stagingAllocCI,
                                 &fr.stagingBuffer, &fr.stagingAlloc, &stagingInfo));
        fr.mappedStaging = stagingInfo.pMappedData;

        // Segmentation readback, at the supersampled size: labels are rendered
        // there so thin geometry survives rasterisation, then reduced to output
        // size on the CPU by class priority.
        VkBufferCreateInfo segStagingCI = stagingCI;
        segStagingCI.size = (VkDeviceSize)w * h * 4;
        VmaAllocationInfo segStagingInfo;
        VK_CHECK(vmaCreateBuffer(ctx.allocator, &segStagingCI, &stagingAllocCI,
                                 &fr.segStagingBuffer, &fr.segStagingAlloc,
                                 &segStagingInfo));
        fr.mappedSegStaging = segStagingInfo.pMappedData;

        // Depth readback: one float per pixel, at the supersampled size.
        VkBufferCreateInfo segDepthStagingCI = stagingCI;
        segDepthStagingCI.size = (VkDeviceSize)w * h * sizeof(float);
        VmaAllocationInfo segDepthStagingInfo;
        VK_CHECK(vmaCreateBuffer(ctx.allocator, &segDepthStagingCI, &stagingAllocCI,
                                 &fr.segDepthStagingBuffer, &fr.segDepthStagingAlloc,
                                 &segDepthStagingInfo));
        fr.mappedSegDepthStaging = segDepthStagingInfo.pMappedData;

        // UBO buffer (host-visible, persistently mapped)
        VkBufferCreateInfo uboCI = {};
        uboCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        uboCI.size = sizeof(SceneUBO);
        uboCI.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

        VmaAllocationCreateInfo uboAllocCI = {};
        uboAllocCI.usage = VMA_MEMORY_USAGE_AUTO;
        uboAllocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                           VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo uboInfo;
        VK_CHECK(vmaCreateBuffer(ctx.allocator, &uboCI, &uboAllocCI,
                                 &fr.uboBuffer, &fr.uboAlloc, &uboInfo));
        fr.mappedUBO = uboInfo.pMappedData;

        // Allocate UBO descriptor set
        fr.uboDescSet = rp.allocateUBODescriptorSet(ctx, fr.uboBuffer);
    }
}

const void* FramePipeline::waitAndReadback(VkContext& ctx) {
    auto& fr = current();
    VK_CHECK(vkWaitForFences(ctx.device, 1, &fr.fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(ctx.device, 1, &fr.fence));

    // Invalidate host cache for staging buffer
    vmaInvalidateAllocation(ctx.allocator, fr.stagingAlloc, 0, VK_WHOLE_SIZE);

    return fr.mappedStaging;
}

void FramePipeline::advance() {
    currentFrame = (currentFrame + 1) % FRAMES_IN_FLIGHT;
}

void FramePipeline::reapCompletedWrites() {
    // get() on each finished future so a failed write surfaces here rather
    // than at shutdown.
    auto done = [](std::future<void>& f) {
        if (f.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            return false;
        f.get();
        return true;
    };
    pendingWrites.erase(std::remove_if(pendingWrites.begin(), pendingWrites.end(), done),
                        pendingWrites.end());
}

void FramePipeline::flush() {
    for (auto& f : pendingWrites) {
        f.get();
    }
    pendingWrites.clear();
}

void FramePipeline::encodeAndWrite(const void* rgbaPixels, uint32_t w, uint32_t h,
                                    const std::string& path, int quality) {
    // Copy pixels (the staging buffer will be reused)
    size_t size = width * height * 4;
    auto pixels = std::make_shared<std::vector<unsigned char>>(size);
    memcpy(pixels->data(), rgbaPixels, size);

    pendingWrites.push_back(pool.submit([pixels, w, h, path, quality]() {
        tjhandle handle = tjInitCompress();
        if (!handle) return;

        unsigned char* jpegBuf = nullptr;
        unsigned long jpegSize = 0;

        // Convert RGBA to RGB for libjpeg-turbo
        std::vector<unsigned char> rgb(w * h * 3);
        const unsigned char* src = pixels->data();
        for (uint32_t i = 0; i < w * h; ++i) {
            rgb[i * 3 + 0] = src[i * 4 + 0];
            rgb[i * 3 + 1] = src[i * 4 + 1];
            rgb[i * 3 + 2] = src[i * 4 + 2];
        }

        int ret = tjCompress2(handle, rgb.data(), w, 0, h, TJPF_RGB,
                              &jpegBuf, &jpegSize, TJSAMP_420, quality, TJFLAG_FASTDCT);
        if (ret == 0 && jpegBuf) {
            FILE* f = fopen(path.c_str(), "wb");
            if (f) {
                fwrite(jpegBuf, 1, jpegSize, f);
                fclose(f);
            }
        }
        tjFree(jpegBuf);
        tjDestroy(handle);
    }));
}

void FramePipeline::writeFile(const std::string& path, const std::string& content) {
    auto contentCopy = std::make_shared<std::string>(content);
    pendingWrites.push_back(pool.submit([path, contentCopy]() {
        std::ofstream f(path);
        f << *contentCopy;
    }));
}

void FramePipeline::destroy(VkContext& ctx) {
    flush();

    for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        auto& fr = frames[i];
        vkDestroyFence(ctx.device, fr.fence, nullptr);
        vkDestroyFramebuffer(ctx.device, fr.framebuffer, nullptr);
        vkDestroyImageView(ctx.device, fr.msaaColorView, nullptr);
        vmaDestroyImage(ctx.allocator, fr.msaaColorImage, fr.msaaColorAlloc);
        vkDestroyImageView(ctx.device, fr.colorView, nullptr);
        vmaDestroyImage(ctx.allocator, fr.colorImage, fr.colorAlloc);
        vmaDestroyImage(ctx.allocator, fr.readbackImage, fr.readbackAlloc);
        vkDestroyImageView(ctx.device, fr.depthView, nullptr);
        vmaDestroyImage(ctx.allocator, fr.depthImage, fr.depthAlloc);
        vmaDestroyBuffer(ctx.allocator, fr.stagingBuffer, fr.stagingAlloc);
        if (fr.segFramebuffer) vkDestroyFramebuffer(ctx.device, fr.segFramebuffer, nullptr);
        if (fr.segView) vkDestroyImageView(ctx.device, fr.segView, nullptr);
        if (fr.segImage) vmaDestroyImage(ctx.allocator, fr.segImage, fr.segAlloc);
        if (fr.segDepthView) vkDestroyImageView(ctx.device, fr.segDepthView, nullptr);
        if (fr.segDepthImage) vmaDestroyImage(ctx.allocator, fr.segDepthImage, fr.segDepthAlloc);
        if (fr.segStagingBuffer)
            vmaDestroyBuffer(ctx.allocator, fr.segStagingBuffer, fr.segStagingAlloc);
        if (fr.segDepthStagingBuffer)
            vmaDestroyBuffer(ctx.allocator, fr.segDepthStagingBuffer, fr.segDepthStagingAlloc);
        vmaDestroyBuffer(ctx.allocator, fr.uboBuffer, fr.uboAlloc);
    }
}

} // namespace dart
