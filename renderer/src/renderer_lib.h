#pragma once

#include "image_index.h"
#include <random>
#include "vk_context.h"
#include "render_pass.h"
#include "scene.h"
#include "accel_structure.h"
#include "randomizer.h"
#include "annotation.h"
#include "frame_pipeline.h"
#include "constants.h"

#include <string>
#include <vector>
#include <filesystem>

namespace dart {

// Record command buffer for one frame (shared between CLI and library)
// accel may be nullptr when RT is disabled
// bgTextureDS: when non-null, binds this texture for the background pass
//              (texture mode); when null, uses Perlin noise background
// transparent: skip the background pass and leave alpha 0 behind the scene
void recordFrame(
    VkCommandBuffer cmd,
    FrameResources& fr,
    RenderPass& rp,
    Scene& scene,
    const FrameState& state,
    uint32_t width, uint32_t height,
    VkContext* ctx = nullptr,
    AccelStructure* accel = nullptr,
    bool transparent = false,
    int frameIndex = 0,
    VkDescriptorSet bgTextureDS = VK_NULL_HANDLE);

/// Fill the frame's SceneUBO from `state` and flush it to the device.
void writeSceneUBO(VkContext& ctx, FrameResources& fr, const Scene& scene,
                   const FrameState& state);

/// Background photographs, streamed from a directory through a fixed pool of
/// textures, one replaced per frame so a long run sees far more photos than
/// fit in memory at once.
///
/// Shared by the Python Renderer and dartboard_gen so both draw backgrounds
/// the same way. Each texture carries a full mip chain: the same photo is the
/// environment probe the surface shaders read at high LODs.
class BackgroundPool {
public:
    static constexpr int kPoolSize = 16;

    /// List the photos in `dir` and load kPoolSize of them at random, resized
    /// to width x height. `seed` fixes which photos are drawn and when, so
    /// renderers with different seeds see different backgrounds. Leaves the
    /// pool empty, with a warning, if `dir` is not a directory or holds no
    /// images.
    void init(VkContext& ctx, RenderPass& rp, const std::string& dir,
              int width, int height, uint32_t seed);
    void destroy(VkContext& ctx);

    bool empty() const { return textures_.empty(); }

    /// Start of recording for pipeline slot `slot`, after waiting on its
    /// fence and before anything that samples a background. Releases what
    /// the slot's previous frame held, then replaces one texture that no
    /// in-flight frame references, recording the upload into `cmd`.
    void beginFrame(VkContext& ctx, RenderPass& rp, VkCommandBuffer cmd, int slot);

    /// This frame's background: a random pool photo about 70% of the time,
    /// VK_NULL_HANDLE (Perlin noise) otherwise. The returned set is held for
    /// `slot` until its next beginFrame, which keeps it out of rotation while
    /// the frame is in flight.
    VkDescriptorSet pick(int slot);

private:
    bool loadNext(std::vector<unsigned char>& rgb);
    static void destroyTexture(VkContext& ctx, Texture& tex);

    ImageIndex index_;
    std::mt19937 rng_;
    int width_ = 0, height_ = 0;
    std::vector<Texture> textures_;
    int rotateIdx_ = 0;

    /// Per slot: the background set the slot's in-flight frame samples.
    VkDescriptorSet inUse_[FramePipeline::FRAMES_IN_FLIGHT] = {};

    /// Per slot: the staging buffer of the upload recorded into the slot's
    /// command buffer, freed once that buffer has executed.
    struct PendingUpload {
        VkBuffer      staging      = VK_NULL_HANDLE;
        VmaAllocation stagingAlloc = VK_NULL_HANDLE;
    };
    PendingUpload pendingUploads_[FramePipeline::FRAMES_IN_FLIGHT] = {};
};

class Renderer {
public:
    Renderer(const std::string& assetDir, int width, int height, uint32_t seed,
             int gpuIndex = -1, const std::string& bgImageDir = "",
             bool skillPlacement = true,
             std::vector<float> dartCountWeights = {},
             float groupingProb = 0.55f,
             float tightProb = 0.55f,
             float panMaxDeg = 60.0f,
             float tiltMaxDeg = 48.0f,
             float envRoomScale = 1.0f,
             const std::string& gpuUuid = "");
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    struct FrameResult {
        std::vector<uint8_t> pixels;    // H*W*channels
        int channels;                    // 3 (RGB with bg) or 4 (RGBA transparent)
        FrameAnnotation annotation;

        /// Per-pixel labels for dense pretraining, H*W*2: channel 0 is
        /// dart::SegClass, channel 1 is the dart instance (0 = not a dart,
        /// else dart index + 1). Empty when the segmentation pass is off.
        ///
        /// Two channels rather than the attachment's four: the spare two carry
        /// nothing, and at 512x512 they would double the transfer for no
        /// information.
        std::vector<uint8_t> segIds;

        /// Window-space depth from the segmentation pass, H*W floats, pixel
        /// registered with segIds. Raw rather than linearised: turning it into
        /// a height above the board plane needs the view matrix and near/far,
        /// which the annotation carries, and doing that here would bake one
        /// interpretation into the renderer.
        std::vector<float> depth;

        /// Which of the pixel's kSupersample x kSupersample subsamples segIds
        /// and depth were taken from, as dy * kSupersample + dx, H*W. The
        /// depth belongs to that subsample's centre, not the pixel's, so
        /// unprojecting it needs the offset.
        std::vector<uint8_t> segSubpixel;
    };

    FrameResult renderFrame();

    int width() const { return width_; }
    int height() const { return height_; }
    bool hasBgTextures() const { return !bgPool_.empty(); }

private:
    void submitFrame();  // randomize, annotate, record, submit to current slot

    VkContext ctx_;
    RenderPass rp_;
    Scene scene_;
    AccelStructure accel_;
    FramePipeline pipeline_;
    Randomizer randomizer_;
    BackgroundPool bgPool_;
    int width_, height_;
    int frameId_ = 0;

    // Double-buffered: store pending annotation per pipeline slot
    FrameAnnotation pendingAnnotations_[FramePipeline::FRAMES_IN_FLIGHT];
};

} // namespace dart
