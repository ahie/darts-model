#include "renderer_lib.h"

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

struct Args {
    int numFrames = 1000;
    std::string outputDir = "output";
    std::string assetDir = DEFAULT_ASSET_DIR;
    uint32_t seed = 42;
    int width = 1024;
    int height = 1024;
    int threads = 4;
    int gpuIndex = -1;
    std::string gpuUuid;
    std::string bgImageDir;
};

static Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--num_frames") == 0 && i + 1 < argc)
            args.numFrames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--output_dir") == 0 && i + 1 < argc)
            args.outputDir = argv[++i];
        else if (strcmp(argv[i], "--asset_dir") == 0 && i + 1 < argc)
            args.assetDir = argv[++i];
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            args.seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc)
            args.width = atoi(argv[++i]);
        else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc)
            args.height = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            args.threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc)
            args.gpuIndex = atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpu_uuid") == 0 && i + 1 < argc)
            args.gpuUuid = argv[++i];
        else if (strcmp(argv[i], "--bg_image_dir") == 0 && i + 1 < argc)
            args.bgImageDir = argv[++i];
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: dartboard_gen [options]\n"
                   "  --num_frames N   Number of frames to generate (default: 1000)\n"
                   "  --output_dir DIR Output directory (default: output)\n"
                   "  --asset_dir DIR  Numeral GLBs and decal atlas (default: " DEFAULT_ASSET_DIR ")\n"
                   "  --seed N         Random seed (default: 42)\n"
                   "  --width N        Image width (default: 1024)\n"
                   "  --height N       Image height (default: 1024)\n"
                   "  --threads N      Thread pool size (default: 4)\n"
                   "  --gpu N          Vulkan device index (default: auto-select)\n"
                   "  --gpu_uuid U     GPU UUID, as nvidia-smi -L or torch.cuda report it\n"
                   "  --bg_image_dir D Background photos, streamed as in training\n"
                   "                   (default: none, Perlin-noise backgrounds)\n");
            exit(0);
        }
    }
    return args;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);

    printf("Dartboard Generator — Vulkan Headless Renderer\n");
    printf("  Frames: %d, Resolution: %dx%d, Seed: %u\n",
           args.numFrames, args.width, args.height, args.seed);
    printf("  Output: %s, Assets: %s\n", args.outputDir.c_str(), args.assetDir.c_str());
    if (!args.bgImageDir.empty())
        printf("  Backgrounds: %s\n", args.bgImageDir.c_str());

    // Create output directories
    fs::create_directories(args.outputDir + "/rgb");
    fs::create_directories(args.outputDir + "/annotations");

    // Initialize Vulkan
    dart::VkContext ctx;
    ctx.init(args.gpuIndex, args.gpuUuid);

    dart::RenderPass rp;
    rp.init(ctx, args.width * dart::kSupersample, args.height * dart::kSupersample);

    // Load scene
    dart::Scene scene;
    scene.loadAssets(ctx, rp, args.assetDir);

    // Acceleration structure (RT only)
    dart::AccelStructure accel;
    if (ctx.rtEnabled) {
        accel.init(ctx, scene, rp.descriptorPool, rp.descSetLayout2);
    }

    // Frame pipeline
    dart::FramePipeline pipeline(args.threads);
    pipeline.init(ctx, rp, args.width, args.height);

    // Background photos, the same streaming pool the Python Renderer uses.
    // Without a directory every frame gets a Perlin-noise background.
    dart::BackgroundPool bgPool;
    if (!args.bgImageDir.empty()) {
        bgPool.init(ctx, rp, args.bgImageDir, args.width, args.height, args.seed);
    }

    // Randomizer
    dart::Randomizer randomizer(args.seed);

    auto startTime = std::chrono::high_resolution_clock::now();

    // --- Main generation loop ---
    // Double-buffered: GPU renders frame N while CPU processes frame N-2
    for (int frameId = 0; frameId < args.numFrames + dart::FramePipeline::FRAMES_IN_FLIGHT; ++frameId) {
        // Wait for oldest in-flight frame and read back its pixels
        const void* pixels = pipeline.waitAndReadback(ctx);

        // Dispatch JPEG encode + annotation write for the completed frame
        int completedFrame = frameId - dart::FramePipeline::FRAMES_IN_FLIGHT;
        if (completedFrame >= 0 && completedFrame < args.numFrames && pixels) {
            char imgPath[512];
            snprintf(imgPath, sizeof(imgPath), "%s/rgb/%06d.jpg",
                     args.outputDir.c_str(), completedFrame);

            pipeline.encodeAndWrite(pixels, args.width, args.height, imgPath);
        }
        pipeline.reapCompletedWrites();

        // If we still have frames to render
        if (frameId < args.numFrames) {
            // Randomize scene + compute annotations (pure CPU), redrawing
            // until every dart has both points inside the frame
            dart::FrameAnnotation ann;
            dart::FrameState state = dart::randomizeWithDartsInFrame(
                randomizer, frameId, dart::RING_RADII_BU,
                scene.boardZ, dart::H_APERTURE_MM,
                args.width, args.height,
                scene.dartGeometries, ann);

            // Retire one design and generate a replacement, every so often.
            // After the randomizer, so the variant this frame chose is known
            // and can be spared.
            dart::rollDartVariant(ctx, scene, accel, frameId,
                                  state.dartVariant);

            // Write annotation JSON
            {
                char annPath[512];
                snprintf(annPath, sizeof(annPath), "%s/annotations/%06d.json",
                         args.outputDir.c_str(), frameId);
                std::string jsonStr = dart::annotationToJson(ann).dump(2);
                pipeline.writeFile(annPath, jsonStr);
            }

            auto& fr = pipeline.current();
            dart::writeSceneUBO(ctx, fr, scene, state);

            // Record and submit command buffer
            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(fr.commandBuffer, &beginInfo);

            // Swap one background photo; its upload is recorded ahead of
            // the render pass that may sample it.
            bgPool.beginFrame(ctx, rp, fr.commandBuffer, pipeline.currentFrame);
            VkDescriptorSet bgDS = bgPool.pick(pipeline.currentFrame);

            dart::recordFrame(fr.commandBuffer, fr, rp, scene, state,
                              args.width * dart::kSupersample,
                              args.height * dart::kSupersample,
                              ctx.rtEnabled ? &ctx : nullptr,
                              ctx.rtEnabled ? &accel : nullptr,
                              /*transparent=*/false,
                              pipeline.currentFrame,
                              bgDS);

            vkEndCommandBuffer(fr.commandBuffer);

            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &fr.commandBuffer;

            vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, fr.fence);
        }

        pipeline.advance();

        // Progress
        if (completedFrame >= 0 && (completedFrame + 1) % 100 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - startTime).count();
            double fps = (completedFrame + 1) / elapsed;
            printf("  [%d/%d] %.1f fps\n", completedFrame + 1, args.numFrames, fps);
        }
    }

    // Flush pending writes
    pipeline.flush();

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(endTime - startTime).count();
    printf("Done: %d frames in %.2f s (%.1f fps)\n",
           args.numFrames, totalTime, args.numFrames / totalTime);

    // Write manifest
    {
        nlohmann::json manifest;
        manifest["num_frames"] = args.numFrames;
        manifest["resolution"] = {args.width, args.height};
        manifest["seed"] = args.seed;
        manifest["generator"] = "vulkan_renderer";

        std::string manifestPath = args.outputDir + "/manifest.json";
        std::ofstream f(manifestPath);
        f << manifest.dump(2);
    }

    // Cleanup
    vkDeviceWaitIdle(ctx.device);
    bgPool.destroy(ctx);
    pipeline.destroy(ctx);
    if (ctx.rtEnabled) accel.destroy(ctx);
    scene.destroy(ctx);
    rp.destroy(ctx);
    ctx.destroy();

    return 0;
}
