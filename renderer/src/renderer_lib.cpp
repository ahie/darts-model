#include "renderer_lib.h"
#include "board_transforms.h"

#include <stb_image.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>

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

namespace fs = std::filesystem;

// recordFrame reduces the supersampled image to the output with one linear
// blit, which is an exact box filter only at a 2x reduction. Another factor
// needs a chain of halving blits (or a compute downsample) there.
static_assert(kSupersample == 2,
              "recordFrame's single-blit downsample assumes kSupersample == 2");

// ---------------------------------------------------------------------------
// Bilinear image resize (CPU)
// ---------------------------------------------------------------------------

static std::vector<unsigned char> resizeImageBilinear(
    const unsigned char* src, int srcW, int srcH,
    int dstW, int dstH, int channels)
{
    std::vector<unsigned char> dst(dstW * dstH * channels);
    for (int y = 0; y < dstH; y++) {
        float fy = (y + 0.5f) * srcH / (float)dstH - 0.5f;
        int y0 = std::max((int)fy, 0);
        int y1 = std::min(y0 + 1, srcH - 1);
        float wy = fy - y0;
        for (int x = 0; x < dstW; x++) {
            float fx = (x + 0.5f) * srcW / (float)dstW - 0.5f;
            int x0 = std::max((int)fx, 0);
            int x1 = std::min(x0 + 1, srcW - 1);
            float wx = fx - x0;
            for (int c = 0; c < channels; c++) {
                float v = src[(y0 * srcW + x0) * channels + c] * (1-wx) * (1-wy)
                        + src[(y0 * srcW + x1) * channels + c] * wx * (1-wy)
                        + src[(y1 * srcW + x0) * channels + c] * (1-wx) * wy
                        + src[(y1 * srcW + x1) * channels + c] * wx * wy;
                dst[(y * dstW + x) * channels + c] = (unsigned char)(v + 0.5f);
            }
        }
    }
    return dst;
}

// ---------------------------------------------------------------------------
// Record command buffer for one frame
// ---------------------------------------------------------------------------

void recordFrame(
    VkCommandBuffer cmd,
    FrameResources& fr,
    RenderPass& rp,
    Scene& scene,
    const FrameState& state,
    uint32_t width, uint32_t height,
    VkContext* ctx,
    AccelStructure* accel,
    bool transparent,
    int frameIndex,
    VkDescriptorSet bgTextureDS)
{
    // Build/update TLAS before render pass (needs to be outside render pass)
    if (accel && ctx) {
        accel->updateTLAS(*ctx, cmd, scene, state, frameIndex);
    }

    // Transparent black. In transparent mode that is the background the
    // caller composites over; otherwise the fullscreen background pass below
    // writes every pixel, so the clear value never reaches the output.
    VkClearValue clearValues[2] = {};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpBegin = {};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = rp.renderPass;
    rpBegin.framebuffer = fr.framebuffer;
    rpBegin.renderArea.extent = {width, height};
    rpBegin.clearValueCount = 2;
    rpBegin.pClearValues = clearValues;

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    // Background: fullscreen triangle (skip for transparent mode)
    if (!transparent) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp.bgPipeline);

        // Bind bg texture descriptor set (required by pipeline layout)
        // In Perlin mode, bind dummy white texture -- it won't be sampled
        VkDescriptorSet bgDS = bgTextureDS != VK_NULL_HANDLE
            ? bgTextureDS
            : scene.textures[scene.dummyWhiteTextureIndex].descriptorSet;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp.bgPipelineLayout,
                                0, 1, &bgDS, 0, nullptr);

        // bgColorA/B must be linear: the SRGB attachment encodes on store.
        BgPushConstants bgPc;
        bgPc.colorA = glm::vec4(state.bgColorA,
                                bgTextureDS != VK_NULL_HANDLE ? 1.0f : 0.0f);
        bgPc.colorB = glm::vec4(state.bgColorB, 0.0f);
        bgPc.params = glm::vec4(state.bgNoiseScale, state.bgNoiseOffsetX, state.bgNoiseOffsetY, state.bgIntensity);
        vkCmdPushConstants(cmd, rp.bgPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(bgPc), &bgPc);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    // Scene geometry
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp.pipeline);

    // Bind UBO descriptor set (set 0)
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp.pipelineLayout,
                            0, 1, &fr.uboDescSet, 0, nullptr);

    // Bind TLAS descriptor set (set 3) when RT is enabled
    if (accel) {
        VkDescriptorSet tlasDS = accel->currentDescSet(frameIndex);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp.pipelineLayout,
                                3, 1, &tlasDS, 0, nullptr);
    }

    // Environment probe: the same background texture composited behind the
    // scene, so reflections agree with the backdrop rather than inventing one.
    // The TLAS occupies set 3 under RT, so the environment moves to set 4
    // there. In Perlin mode there is no photo to reflect; the dummy white
    // texture stands in as a neutral studio environment.
    {
        VkDescriptorSet envDS = bgTextureDS != VK_NULL_HANDLE
            ? bgTextureDS
            : scene.textures[scene.dummyEnvTextureIndex].descriptorSet;
        uint32_t envSet = accel ? 4u : 3u;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp.pipelineLayout,
                                envSet, 1, &envDS, 0, nullptr);
    }

    // Decal atlas, one set past the environment probe. Always present:
    // Scene::loadAssets aborts without one.
    {
        VkDescriptorSet decalDS =
            scene.textures[scene.decalAtlasTextureIndex].descriptorSet;
        uint32_t decalSet = accel ? 5u : 4u;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp.pipelineLayout,
                                decalSet, 1, &decalDS, 0, nullptr);
    }

    // Albedo (set 1) and normal map (set 2). Nothing drawn here is textured
    // -- the face is evaluated analytically and everything else takes a flat
    // colour -- but the layout requires both sets, so the neutral dummies
    // are bound once for the whole pass.
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp.pipelineLayout,
                            1, 1, &scene.textures[scene.dummyWhiteTextureIndex].descriptorSet,
                            0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp.pipelineLayout,
                            2, 1, &scene.textures[scene.dummyFlatNormalTextureIndex].descriptorSet,
                            0, nullptr);

    // The same projection the annotations use (cameraProjection), plus the
    // Vulkan Y-flip.
    glm::mat4 proj = cameraProjection(state.camera.focalLength, H_APERTURE_MM,
                                      width, height, kNearPlane, kFarPlane);
    proj[1][1] *= -1.0f;

    glm::mat4 view = state.camera.viewMatrix;

    // Lambda to draw a mesh
    auto drawMesh = [&](const Mesh& mesh, const glm::mat4& modelMatrix,
                        uint32_t drawMode, glm::vec3 color) {
        if (mesh.indexCount == 0) return;

        glm::mat4 mvp = proj * view * modelMatrix;

        PushConstants pc = {};
        pc.mvp = mvp;
        pc.modelCol0 = glm::vec4(modelMatrix[0][0], modelMatrix[0][1], modelMatrix[0][2], modelMatrix[3][0]);
        pc.modelCol1 = glm::vec4(modelMatrix[1][0], modelMatrix[1][1], modelMatrix[1][2], modelMatrix[3][1]);
        pc.modelCol2 = glm::vec4(modelMatrix[2][0], modelMatrix[2][1], modelMatrix[2][2], modelMatrix[3][2]);
        pc.drawMode = drawMode;
        pc.colorR = color.r;
        pc.colorG = color.g;
        pc.colorB = color.b;

        vkCmdPushConstants(cmd, rp.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer, &offset);
        vkCmdBindIndexBuffer(cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
    };

    // Board rotation matrix (numbers + face together)
    glm::mat4 boardRot = glm::rotate(glm::mat4(1.0f), state.boardRotation, glm::vec3(0, 0, 1));
    // Additional face-only rotation (breaks texture/number correlation)
    glm::mat4 boardFaceRot = glm::rotate(glm::mat4(1.0f), state.boardFaceRotation, glm::vec3(0, 0, 1));

    // 1. Board face (rotated by both boardRot and boardFaceRot)
    {
        glm::mat4 model = boardRot * boardFaceRot * scene.boardFace.baseTransform;
        // Bit 29 marks the one surface printed marks are composited onto. The
        // wires, numerals and darts must not receive them.
        uint32_t mode = packDrawMode(false, false, 0.0f, Scene::kBoardFaceRoughness)
                      | (1u << 29);
        drawMesh(scene.boardFace, model, mode, glm::vec3(1.0f));
    }

    // 1b. Spider (wire frame). Belongs to the playing surface, so it takes the
    // same rotation as the face -- the wires separate the beds and must stay
    // locked to them, independent of how the number ring is rotated.
    if (const Mesh* spider = scene.spiderFor(state.spiderVariant)) {
        glm::mat4 model = boardRot * boardFaceRot * spider->baseTransform;
        drawMesh(*spider, model,
                 packDrawMode(false, false, state.wireMetallic, state.wireRoughness),
                 state.wireColor);
    }

    // 2. Number meshes (PBR, rotated + scaled)
    if (state.fontVariant < (int)scene.numberVariants.size()) {
        auto& numbers = scene.numberVariants[state.fontVariant];
        glm::mat4 scaleM = glm::scale(glm::mat4(1.0f), glm::vec3(state.numberScale));
        float numMetallic  = state.numberMetallic ? 1.0f : 0.0f;
        float numRoughness = state.numberMetallic ? 0.3f : 0.7f;
        float numGray      = state.numberMetallic ? 0.75f : 1.0f;
        for (auto& numMesh : numbers) {
            glm::vec3 pos = glm::vec3(numMesh.baseTransform[3]);
            glm::mat4 localXform = numMesh.baseTransform;
            localXform[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

            glm::mat4 model = numeralModelMatrix(boardRot, pos, scaleM, localXform,
                                                 state.numberRingOffset);
            drawMesh(numMesh, model,
                     packDrawMode(false, false, numMetallic, numRoughness),
                     glm::vec3(numGray));
        }
    }

    // 3. Darts (randomized materials per part)
    const Scene::DartVariant* dv = scene.dartFor(state.dartVariant);
    for (int i = 0; dv != nullptr && i < state.numDarts; ++i) {
        const DartMaterial& dm = state.dartMaterials[i];
        for (const auto& dartMesh : dv->meshes) {
            uint32_t mode;
            glm::vec3 color;
            if (dartMesh.dartPart == DartPart::Flight) {
                // Translucent flights are deferred to a second pass so they
                // blend over whatever ends up behind them.
                if (dm.flightAlpha < 0.999f) continue;
                mode = packDrawMode(false, false, 0.0f, dm.flightRoughness);
                color = dm.flightColor;
            } else if (dartMesh.dartPart == DartPart::Point) {
                mode = packDrawMode(false, false, dm.pointMetallic, dm.pointRoughness);
                color = dm.pointColor;
            } else {
                mode = packDrawMode(false, false, dm.metalMetallic, dm.metalRoughness);
                color = dm.metalColor;
            }
            drawMesh(dartMesh, state.dartTransforms[i], mode, color);
        }
    }

    // Translucent flights, drawn last so they blend against the finished scene.
    // Blending happens in linear light, since the attachment is SRGB.
    // Depth testing stays on so the board occludes them; depth writes are off
    // so a flight does not hide the dart behind it.
    //
    // Not sorted back-to-front: flights are thin, nearly planar and rarely
    // overlap each other, so the ordering artefact this leaves is smaller than
    // the cost of sorting per frame.
    {
        bool anyBlend = false;
        for (int i = 0; i < state.numDarts && dv != nullptr; ++i)
            if (state.dartMaterials[i].flightAlpha < 0.999f) anyBlend = true;

        if (anyBlend && rp.blendPipeline != VK_NULL_HANDLE) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp.blendPipeline);
            for (int i = 0; i < state.numDarts && dv != nullptr; ++i) {
                const DartMaterial& dm = state.dartMaterials[i];
                if (dm.flightAlpha >= 0.999f) continue;
                for (auto& dartMesh : dv->meshes) {
                    if (dartMesh.dartPart != DartPart::Flight) continue;
                    drawMesh(dartMesh, state.dartTransforms[i],
                             // bit 28 marks translucent; the low seven bits
                             // carry the alpha, which the push constants have
                             // no float field for.
                             packDrawMode(false, false, 0.0f, dm.flightRoughness)
                                 | (1u << 28)
                                 | (uint32_t)(glm::clamp(dm.flightAlpha, 0.0f, 1.0f)
                                              * 127.0f + 0.5f),
                             dm.flightColor);
                }
            }
        }
    }

    vkCmdEndRenderPass(cmd);

    // Filter the supersampled resolve down to the output size.
    //
    // One linear blit. At exactly 2x reduction a linear filter samples the
    // four covered texels with equal weight, so the blit is an exact 2x2 box
    // average; at larger factors it would skip texels, hence the assert at
    // the top of this file. Both images are SRGB, and a blit between SRGB
    // images decodes, filters in linear and re-encodes, so the average is
    // taken in linear light.
    //
    // The render pass's outgoing dependency makes the resolve writes visible
    // to this transfer and leaves colorImage in TRANSFER_SRC_OPTIMAL.
    const uint32_t outW = width / kSupersample;
    const uint32_t outH = height / kSupersample;
    {
        auto barrier = [&](VkImage img, VkImageLayout from, VkImageLayout to,
                           VkAccessFlags srcA, VkAccessFlags dstA,
                           VkPipelineStageFlags srcS) {
            VkImageMemoryBarrier b = {};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout = from; b.newLayout = to;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = img;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            b.srcAccessMask = srcA; b.dstAccessMask = dstA;
            vkCmdPipelineBarrier(cmd, srcS, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                 0, nullptr, 0, nullptr, 1, &b);
        };

        // The previous contents are discarded. That frame's copy-out
        // finished before this slot's fence was signalled.
        barrier(fr.readbackImage, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

        VkImageBlit blit = {};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {(int32_t)width, (int32_t)height, 1};
        blit.dstOffsets[1] = {(int32_t)outW, (int32_t)outH, 1};
        vkCmdBlitImage(cmd, fr.colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       fr.readbackImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, VK_FILTER_LINEAR);

        barrier(fr.readbackImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    // Copy the output-sized image to the staging buffer. A copy moves raw
    // texels, so the host receives sRGB-encoded bytes.
    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {outW, outH, 1};

    vkCmdCopyImageToBuffer(cmd, fr.readbackImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           fr.stagingBuffer, 1, &region);

    // --- Segmentation pass: per-pixel class + instance ids ---
    //
    // Re-draws the same geometry at the supersampled resolution with no MSAA.
    // Separate rather than an extra attachment on the main pass because ids
    // must not be averaged, and the main pass both resolves multisamples and
    // box-filters the supersampled result. Reduced to output size on readback
    // in renderFrame().
    if (rp.segPipeline != VK_NULL_HANDLE && fr.segFramebuffer != VK_NULL_HANDLE) {
        VkClearValue segClears[2] = {};
        segClears[0].color.uint32[0] = (uint32_t)SegClass::Background;
        segClears[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo segBegin = {};
        segBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        segBegin.renderPass = rp.segRenderPass;
        segBegin.framebuffer = fr.segFramebuffer;
        segBegin.renderArea.extent = {width, height};
        segBegin.clearValueCount = 2;
        segBegin.pClearValues = segClears;

        vkCmdBeginRenderPass(cmd, &segBegin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp.segPipeline);

        auto drawIds = [&](const Mesh& mesh, const glm::mat4& modelMatrix,
                           SegClass cls, uint32_t instance) {
            if (mesh.indexCount == 0) return;
            SegPushConstants spc = {};
            spc.mvp = proj * view * modelMatrix;
            spc.segClass = (uint32_t)cls;
            spc.segInstance = instance;
            vkCmdPushConstants(cmd, rp.segPipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(spc), &spc);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer, &offset);
            vkCmdBindIndexBuffer(cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
        };

        // Transforms are recomputed rather than reused: the colour pass builds
        // each model matrix inside its own scope. They must match exactly, or
        // the labels will not line up with the pixels they describe.
        drawIds(scene.boardFace,
                boardRot * boardFaceRot * scene.boardFace.baseTransform,
                SegClass::BoardFace, 0);

        if (const Mesh* segSpider = scene.spiderFor(state.spiderVariant)) {
            drawIds(*segSpider, boardRot * boardFaceRot * segSpider->baseTransform,
                    SegClass::Wire, 0);
        }

        if (state.fontVariant < (int)scene.numberVariants.size()) {
            auto& segNumbers = scene.numberVariants[state.fontVariant];
            glm::mat4 segScaleM = glm::scale(glm::mat4(1.0f), glm::vec3(state.numberScale));
            for (auto& numMesh : segNumbers) {
                glm::vec3 pos = glm::vec3(numMesh.baseTransform[3]);
                glm::mat4 localXform = numMesh.baseTransform;
                localXform[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
                drawIds(numMesh,
                        numeralModelMatrix(boardRot, pos, segScaleM, localXform,
                                           state.numberRingOffset),
                        SegClass::Numerals, 0);
            }
        }

        // Darts. Translucent flights are drawn here too: for a label the right
        // answer is the flight, not what shows through it.
        for (int i = 0; i < state.numDarts && dv != nullptr; ++i) {
            for (const auto& dartMesh : dv->meshes) {
                SegClass cls = SegClass::DartMetal;
                if (dartMesh.dartPart == DartPart::Flight)      cls = SegClass::DartFlight;
                else if (dartMesh.dartPart == DartPart::Point)  cls = SegClass::DartPoint;
                drawIds(dartMesh, state.dartTransforms[i], cls, (uint32_t)(i + 1));
            }
        }

        vkCmdEndRenderPass(cmd);

        VkBufferImageCopy segRegion = {};
        segRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        segRegion.imageSubresource.layerCount = 1;
        segRegion.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(cmd, fr.segImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               fr.segStagingBuffer, 1, &segRegion);

        if (fr.segDepthStagingBuffer != VK_NULL_HANDLE) {
            VkBufferImageCopy dRegion = segRegion;
            dRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            vkCmdCopyImageToBuffer(cmd, fr.segDepthImage,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   fr.segDepthStagingBuffer, 1, &dRegion);
        }
    }

    // Make every staging copy above available to the host. The fence wait
    // that precedes the CPU read orders execution, but only this barrier
    // makes the transfer writes visible to host reads.
    {
        VkMemoryBarrier mb = {};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0,
                             1, &mb, 0, nullptr, 0, nullptr);
    }
}

// ---------------------------------------------------------------------------
// Scene UBO
// ---------------------------------------------------------------------------

void writeSceneUBO(VkContext& ctx, FrameResources& fr, const Scene& scene,
                   const FrameState& state) {
    SceneUBO ubo;
    ubo.lightPos = glm::vec4(state.lightPos, state.lightIntensity);
    // Camera position in world space (worldMatrix column 3 = eye position)
    ubo.cameraPos = glm::vec4(
        state.camera.worldMatrix[3][0],
        state.camera.worldMatrix[3][1],
        state.camera.worldMatrix[3][2],
        state.ambient);
    ubo.lightColor = glm::vec4(state.lightColor, 0.0f);
    ubo.material = glm::vec4(state.normalStrength, state.envIntensity,
                             state.contactAOStrength, state.shadowStrength);
    ubo.lightRing = glm::vec4((float)state.lightRingMode, state.lightRingRadius,
                              state.lightRingZ, 0.0f);
    {
        // Must match the board face's TLAS instance transform exactly, or a
        // reflection resolves the hit to the wrong bed.
        glm::mat4 bRot = glm::rotate(glm::mat4(1.0f), state.boardRotation,
                                     glm::vec3(0, 0, 1));
        glm::mat4 fRot = glm::rotate(glm::mat4(1.0f), state.boardFaceRotation,
                                     glm::vec3(0, 0, 1));
        fillBoardFaceUBO(ubo, bRot * fRot * scene.boardFace.baseTransform);
    }
    fillDecalUBO(ubo, state);
    fillSisalUBO(ubo, state);
    fillEnvRoomUBO(ubo, state);
    fillBoardPaletteUBO(ubo, state);
    fillLight2UBO(ubo, state);
    memcpy(fr.mappedUBO, &ubo, sizeof(ubo));
    // HOST_ACCESS_SEQUENTIAL_WRITE memory may be non-coherent; on coherent
    // memory the flush is a no-op.
    vmaFlushAllocation(ctx.allocator, fr.uboAlloc, 0, VK_WHOLE_SIZE);
}

// ---------------------------------------------------------------------------
// Background photo pool
// ---------------------------------------------------------------------------

bool BackgroundPool::loadNext(std::vector<unsigned char>& rgb) {
    // A few attempts, so one unreadable file costs a retry rather than a
    // pool slot.
    for (int attempt = 0; attempt < 8; ++attempt) {
        const std::string path = index_.sample(rng_);
        if (path.empty()) return false;
        int w, h, ch;
        unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &ch, 3);
        if (!pixels) {
            fprintf(stderr, "Warning: failed to load bg image %s\n", path.c_str());
            continue;
        }
        if (w != width_ || h != height_) {
            rgb = resizeImageBilinear(pixels, w, h, width_, height_, 3);
        } else {
            rgb.assign(pixels, pixels + (size_t)w * h * 3);
        }
        stbi_image_free(pixels);
        return true;
    }
    return false;
}

void BackgroundPool::init(VkContext& ctx, RenderPass& rp, const std::string& dir,
                          int width, int height, uint32_t seed) {
    if (!index_.scan(dir)) {
        fprintf(stderr, "Warning: bg_image_dir '%s' is not a directory\n", dir.c_str());
        return;
    }
    if (index_.size() == 0) {
        fprintf(stderr, "Warning: bg_image_dir '%s' holds no images\n", dir.c_str());
        return;
    }
    // Offset from the scene randomizer's seed so the two streams differ.
    rng_.seed(seed ^ 0x6A09E667u);
    width_ = width;
    height_ = height;

    printf("Background pool: %zu images in %s, loading %d ...\n",
           index_.size(), dir.c_str(), kPoolSize);
    std::vector<unsigned char> rgb;
    for (int i = 0; i < kPoolSize; ++i) {
        if (!loadNext(rgb)) break;  // fewer images than pool size
        textures_.push_back(Scene::createTexture(ctx, rp, rgb.data(), width_, height_, 3));
    }
    printf("Background pool: %zu textures loaded\n", textures_.size());
}

void BackgroundPool::beginFrame(VkContext& ctx, RenderPass& rp, VkCommandBuffer cmd,
                                int slot) {
    // The caller has waited on this slot's fence, so whatever it last
    // sampled and uploaded is no longer in use.
    inUse_[slot] = VK_NULL_HANDLE;
    PendingUpload& pending = pendingUploads_[slot];
    if (pending.staging != VK_NULL_HANDLE) {
        vmaDestroyBuffer(ctx.allocator, pending.staging, pending.stagingAlloc);
        pending = {};
    }

    if (textures_.empty()) return;

    // Replace one texture that no in-flight frame references.
    const int poolSize = (int)textures_.size();
    for (int attempt = 0; attempt < poolSize; ++attempt) {
        const int idx = (rotateIdx_ + attempt) % poolSize;
        const VkDescriptorSet ds = textures_[idx].descriptorSet;
        bool busy = false;
        for (int s = 0; s < FramePipeline::FRAMES_IN_FLIGHT; ++s)
            busy |= (inUse_[s] == ds);
        if (busy) continue;

        std::vector<unsigned char> rgb;
        if (!loadNext(rgb)) return;

        rp.freeDescriptorSet(ctx, textures_[idx].descriptorSet);
        destroyTexture(ctx, textures_[idx]);

        // Recorded into this frame's command buffer ahead of the render
        // pass, full mip chain included: the probe is read at LOD 5-6.
        TextureUpload up = recordTextureUpload(ctx, rp, cmd, rgb.data(),
                                               width_, height_, 3);
        textures_[idx] = up.tex;
        pending = {up.staging, up.stagingAlloc};

        rotateIdx_ = (idx + 1) % poolSize;
        return;
    }
}

VkDescriptorSet BackgroundPool::pick(int slot) {
    VkDescriptorSet ds = VK_NULL_HANDLE;
    if (!textures_.empty()) {
        // ~70% photo, ~30% Perlin noise.
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        if (u(rng_) < 0.7f) {
            std::uniform_int_distribution<size_t> slotPick(0, textures_.size() - 1);
            ds = textures_[slotPick(rng_)].descriptorSet;
        }
    }
    inUse_[slot] = ds;
    return ds;
}

void BackgroundPool::destroyTexture(VkContext& ctx, Texture& tex) {
    vkDestroyImageView(ctx.device, tex.imageView, nullptr);
    vmaDestroyImage(ctx.allocator, tex.image, tex.alloc);
    tex = {};
}

void BackgroundPool::destroy(VkContext& ctx) {
    // Descriptor sets go with the render pass's descriptor pool.
    for (auto& tex : textures_) destroyTexture(ctx, tex);
    textures_.clear();
    for (auto& pending : pendingUploads_) {
        if (pending.staging != VK_NULL_HANDLE)
            vmaDestroyBuffer(ctx.allocator, pending.staging, pending.stagingAlloc);
        pending = {};
    }
}

// ---------------------------------------------------------------------------
// Renderer class
// ---------------------------------------------------------------------------

Renderer::Renderer(const std::string& assetDir, int width, int height, uint32_t seed,
                   int gpuIndex, const std::string& bgImageDir,
                   bool skillPlacement, std::vector<float> dartCountWeights,
                   float groupingProb, float tightProb,
                   float panMaxDeg, float tiltMaxDeg, float envRoomScale,
                   const std::string& gpuUuid)
    : pipeline_(0)  // frames are returned in memory; no encode/write threads
    , randomizer_(seed, skillPlacement, std::move(dartCountWeights),
                  groupingProb, tightProb, panMaxDeg, tiltMaxDeg,
                  envRoomScale)
    , width_(width)
    , height_(height)
{
    ctx_.init(gpuIndex, gpuUuid);
    // The render pass owns the viewport, so it is built at the supersampled
    // size; only the readback and the annotations use the output size.
    rp_.init(ctx_, width * kSupersample, height * kSupersample);
    scene_.loadAssets(ctx_, rp_, assetDir);
    if (ctx_.rtEnabled) {
        accel_.init(ctx_, scene_, rp_.descriptorPool, rp_.descSetLayout2);
    }
    pipeline_.init(ctx_, rp_, width, height);

    if (!bgImageDir.empty()) {
        bgPool_.init(ctx_, rp_, bgImageDir, width_, height_, seed);
    }

    // Prime the pipeline: submit FRAMES_IN_FLIGHT frames so the GPU
    // is already working when the first renderFrame() call arrives
    for (int i = 0; i < FramePipeline::FRAMES_IN_FLIGHT; ++i) {
        submitFrame();
        pipeline_.advance();
    }
}

Renderer::~Renderer() {
    vkDeviceWaitIdle(ctx_.device);
    bgPool_.destroy(ctx_);
    pipeline_.destroy(ctx_);
    if (ctx_.rtEnabled) accel_.destroy(ctx_);
    scene_.destroy(ctx_);
    rp_.destroy(ctx_);
    ctx_.destroy();
}

void Renderer::submitFrame() {
    auto& fr = pipeline_.current();
    const int slot = pipeline_.currentFrame;

    // Wait for any previous work on this slot (signaled on first use)
    VK_CHECK(vkWaitForFences(ctx_.device, 1, &fr.fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(ctx_.device, 1, &fr.fence));

    // Begin command buffer -- owns lifetime for both bg upload and render pass
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(fr.commandBuffer, &beginInfo));

    // Swap one background photo (records its upload into the command buffer)
    bgPool_.beginFrame(ctx_, rp_, fr.commandBuffer, slot);

    // Randomize scene + compute annotations (pure CPU), redrawing until every
    // dart has both points inside the frame
    FrameState state = randomizeWithDartsInFrame(
        randomizer_, frameId_, RING_RADII_BU,
        scene_.boardZ, H_APERTURE_MM,
        width_, height_,
        scene_.dartGeometries,
        pendingAnnotations_[slot]);

    // Retire one design and generate a replacement, every so often. After the
    // randomizer, so the variant this frame chose is known and can be spared.
    rollDartVariant(ctx_, scene_, accel_, frameId_, state.dartVariant);

    writeSceneUBO(ctx_, fr, scene_, state);

    const VkDescriptorSet bgDS = bgPool_.pick(slot);

    // Without a photo directory the frame is rendered transparent for the
    // caller to composite; with one, every frame is opaque.
    bool transparent = bgPool_.empty();
    recordFrame(fr.commandBuffer, fr, rp_, scene_, state,
                width_ * kSupersample, height_ * kSupersample,
                ctx_.rtEnabled ? &ctx_ : nullptr,
                ctx_.rtEnabled ? &accel_ : nullptr,
                transparent, slot, bgDS);

    VK_CHECK(vkEndCommandBuffer(fr.commandBuffer));

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &fr.commandBuffer;
    VK_CHECK(vkQueueSubmit(ctx_.graphicsQueue, 1, &submitInfo, fr.fence));

    frameId_++;
}

Renderer::FrameResult Renderer::renderFrame() {
    // Wait for the oldest in-flight frame on the current slot
    auto& fr = pipeline_.current();
    VK_CHECK(vkWaitForFences(ctx_.device, 1, &fr.fence, VK_TRUE, UINT64_MAX));

    // Readback completed RGBA pixels
    vmaInvalidateAllocation(ctx_.allocator, fr.stagingAlloc, 0, VK_WHOLE_SIZE);
    const uint8_t* src = static_cast<const uint8_t*>(fr.mappedStaging);

    // Save the completed annotation before we overwrite the slot
    FrameAnnotation completedAnn = pendingAnnotations_[pipeline_.currentFrame];

    int numPixels = width_ * height_;
    FrameResult result;
    result.annotation = completedAnn;

    if (!bgPool_.empty()) {
        // Strip alpha: RGBA → RGB (background already composited by GPU)
        result.channels = 3;
        result.pixels.resize(numPixels * 3);
        uint8_t* dst = result.pixels.data();
        for (int i = 0; i < numPixels; ++i) {
            dst[i*3+0] = src[i*4+0];
            dst[i*3+1] = src[i*4+1];
            dst[i*3+2] = src[i*4+2];
        }
    } else {
        // Return full RGBA
        result.channels = 4;
        result.pixels.assign(src, src + numPixels * 4);
    }

    // Segmentation ids, reduced from the supersampled render to output size.
    //
    // Not averaged -- averaging labels is meaningless -- and not nearest
    // either, which would drop thin geometry just as rendering at 1x would.
    // Instead the rarest class present in each block wins, and class, instance
    // and depth are all taken from that same subpixel so they stay mutually
    // consistent. Occlusion is already settled by the depth test at subpixel
    // level, so priority only ever arbitrates between things genuinely visible
    // in the block: it cannot resurrect a wire that a dart covers.
    //
    // The bias is deliberate and worth stating: a block that is three quarters
    // board and one quarter wire is labelled wire, so thin classes come out
    // over-represented in area. For a target whose purpose is forcing
    // high-frequency features, a connected wire matters more than its area
    // being right.
    static const int kPriority[(int)SegClass::Count] = {
        7,  // Background -- lowest
        6,  // BoardFace
        1,  // Wire        -- sub-pixel, must survive
        4,  // Numerals
        0,  // DartPoint   -- the tip matters most
        3,  // DartMetal
        5,  // DartFlight
    };
    if (fr.mappedSegStaging != nullptr) {
        vmaInvalidateAllocation(ctx_.allocator, fr.segStagingAlloc, 0, VK_WHOLE_SIZE);
        const uint8_t* segSrc = static_cast<const uint8_t*>(fr.mappedSegStaging);
        const float* depSrc = fr.mappedSegDepthStaging
            ? static_cast<const float*>(fr.mappedSegDepthStaging) : nullptr;
        if (depSrc) {
            vmaInvalidateAllocation(ctx_.allocator, fr.segDepthStagingAlloc, 0, VK_WHOLE_SIZE);
        }
        const int ssW = width_ * kSupersample;
        result.segIds.resize(numPixels * 2);
        if (depSrc) result.depth.resize(numPixels);
        result.segSubpixel.resize(numPixels);
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                int bestPrio = 999, bestIdx = -1, bestSub = 0;
                for (int dy = 0; dy < kSupersample; ++dy) {
                    for (int dx = 0; dx < kSupersample; ++dx) {
                        const int si = (y * kSupersample + dy) * ssW
                                     + (x * kSupersample + dx);
                        const uint8_t c = segSrc[si * 4 + 0];
                        const int prio = (c < (int)SegClass::Count) ? kPriority[c] : 999;
                        if (prio < bestPrio) {
                            bestPrio = prio;
                            bestIdx = si;
                            bestSub = dy * kSupersample + dx;
                        }
                    }
                }
                const int di = y * width_ + x;
                result.segIds[di*2+0] = segSrc[bestIdx * 4 + 0];
                result.segIds[di*2+1] = segSrc[bestIdx * 4 + 1];
                if (depSrc) result.depth[di] = depSrc[bestIdx];
                result.segSubpixel[di] = (uint8_t)bestSub;
            }
        }
    }

    // Submit the next frame to this slot (GPU starts working immediately)
    submitFrame();
    pipeline_.advance();

    return result;
}

} // namespace dart
