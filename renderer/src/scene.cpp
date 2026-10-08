#include "scene.h"

#include "board_mesh.h"
#include "dart_mesh.h"
#include "constants.h"

#include <cgltf.h>
#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <cstdlib>
#include <cstring>
#include <filesystem>

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

// ---------------------------------------------------------------------------
// Buffer upload via staging
// ---------------------------------------------------------------------------

void Scene::stageBuffer(VkContext& ctx, VkCommandBuffer cmd, VkBuffer& buffer,
                        VmaAllocation& alloc, const void* data,
                        VkDeviceSize size, VkBufferUsageFlags usage,
                        std::vector<PendingStage>& pending) {
    VkBufferCreateInfo stagingCI = {};
    stagingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingCI.size = size;
    stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocCI = {};
    stagingAllocCI.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAllocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                           VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer stagingBuf;
    VmaAllocation stagingAlloc;
    VmaAllocationInfo stagingInfo;
    VK_CHECK(vmaCreateBuffer(ctx.allocator, &stagingCI, &stagingAllocCI,
                             &stagingBuf, &stagingAlloc, &stagingInfo));
    memcpy(stagingInfo.pMappedData, data, size);
    vmaFlushAllocation(ctx.allocator, stagingAlloc, 0, VK_WHOLE_SIZE);

    VkBufferCreateInfo bufCI = {};
    bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size = size;
    bufCI.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (ctx.rtEnabled) {
        bufCI.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                     | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    }
    VmaAllocationCreateInfo bufAllocCI = {};
    bufAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VK_CHECK(vmaCreateBuffer(ctx.allocator, &bufCI, &bufAllocCI, &buffer,
                             &alloc, nullptr));

    VkBufferCopy copy = {};
    copy.size = size;
    vkCmdCopyBuffer(cmd, stagingBuf, buffer, 1, &copy);
    pending.push_back({stagingBuf, stagingAlloc});
}

void Scene::uploadBuffer(VkContext& ctx, VkBuffer& buffer, VmaAllocation& alloc,
                         const void* data, VkDeviceSize size, VkBufferUsageFlags usage) {
    // Staging buffer
    VkBufferCreateInfo stagingCI = {};
    stagingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingCI.size = size;
    stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocCI = {};
    stagingAllocCI.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAllocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                           VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer stagingBuf;
    VmaAllocation stagingAlloc;
    VmaAllocationInfo stagingInfo;
    VK_CHECK(vmaCreateBuffer(ctx.allocator, &stagingCI, &stagingAllocCI,
                             &stagingBuf, &stagingAlloc, &stagingInfo));
    memcpy(stagingInfo.pMappedData, data, size);
    vmaFlushAllocation(ctx.allocator, stagingAlloc, 0, VK_WHOLE_SIZE);

    // GPU buffer
    VkBufferCreateInfo bufCI = {};
    bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size = size;
    bufCI.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (ctx.rtEnabled) {
        bufCI.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                     | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    }

    VmaAllocationCreateInfo bufAllocCI = {};
    bufAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VK_CHECK(vmaCreateBuffer(ctx.allocator, &bufCI, &bufAllocCI,
                             &buffer, &alloc, nullptr));

    // Copy
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    VkBufferCopy region = {};
    region.size = size;
    vkCmdCopyBuffer(cmd, stagingBuf, buffer, 1, &region);
    ctx.endSingleTimeCommands(cmd);

    vmaDestroyBuffer(ctx.allocator, stagingBuf, stagingAlloc);
}

// ---------------------------------------------------------------------------
// Texture creation
// ---------------------------------------------------------------------------

TextureUpload recordTextureUpload(VkContext& ctx, RenderPass& rp, VkCommandBuffer cmd,
                                  const unsigned char* pixels, int w, int h,
                                  int channels, bool linear) {
    TextureUpload up;
    Texture& tex = up.tex;
    VkDeviceSize imageSize = (VkDeviceSize)w * h * 4;
    VkFormat format = linear ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R8G8B8A8_SRGB;

    uint32_t mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(w, h)))) + 1;

    // Staging buffer
    VkBufferCreateInfo stagingCI = {};
    stagingCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingCI.size = imageSize;
    stagingCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocCI = {};
    stagingAllocCI.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAllocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                           VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo stagingInfo;
    VK_CHECK(vmaCreateBuffer(ctx.allocator, &stagingCI, &stagingAllocCI,
                             &up.staging, &up.stagingAlloc, &stagingInfo));

    // Expand to RGBA if needed
    if (channels == 4) {
        memcpy(stagingInfo.pMappedData, pixels, imageSize);
    } else {
        auto* dst = static_cast<unsigned char*>(stagingInfo.pMappedData);
        for (int i = 0; i < w * h; ++i) {
            dst[i * 4 + 0] = pixels[i * channels + 0];
            dst[i * 4 + 1] = channels > 1 ? pixels[i * channels + 1] : pixels[i * channels];
            dst[i * 4 + 2] = channels > 2 ? pixels[i * channels + 2] : pixels[i * channels];
            dst[i * 4 + 3] = 255;
        }
    }
    vmaFlushAllocation(ctx.allocator, up.stagingAlloc, 0, VK_WHOLE_SIZE);

    VkImageCreateInfo imgCI = {};
    imgCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCI.imageType = VK_IMAGE_TYPE_2D;
    imgCI.format = format;
    imgCI.extent = {(uint32_t)w, (uint32_t)h, 1};
    imgCI.mipLevels = mipLevels;
    imgCI.arrayLayers = 1;
    imgCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imgCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgCI.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo imgAllocCI = {};
    imgAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VK_CHECK(vmaCreateImage(ctx.allocator, &imgCI, &imgAllocCI,
                            &tex.image, &tex.alloc, nullptr));

    auto barrier = [&](uint32_t level, uint32_t count,
                       VkImageLayout from, VkImageLayout to,
                       VkAccessFlags srcA, VkAccessFlags dstA,
                       VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
        VkImageMemoryBarrier b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = from;
        b.newLayout = to;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = tex.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, count, 0, 1};
        b.srcAccessMask = srcA;
        b.dstAccessMask = dstA;
        vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    // Every level to TRANSFER_DST, then the pixels into level 0.
    barrier(0, mipLevels, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
    vkCmdCopyBufferToImage(cmd, up.staging, tex.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Each level is a linear 2x downsample of the one above. On an SRGB image
    // the blit decodes, filters and re-encodes, so the chain is averaged in
    // linear light.
    int32_t mipW = w, mipH = h;
    for (uint32_t i = 1; i < mipLevels; i++) {
        barrier(i - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        int32_t nextW = mipW > 1 ? mipW / 2 : 1;
        int32_t nextH = mipH > 1 ? mipH / 2 : 1;
        VkImageBlit blit = {};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
        blit.srcOffsets[1] = {mipW, mipH, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
        blit.dstOffsets[1] = {nextW, nextH, 1};
        vkCmdBlitImage(cmd, tex.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, VK_FILTER_LINEAR);

        barrier(i - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        mipW = nextW;
        mipH = nextH;
    }

    // The last level was only ever written.
    barrier(mipLevels - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    VkImageViewCreateInfo viewCI = {};
    viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.image = tex.image;
    viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format = format;
    viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCI.subresourceRange.levelCount = mipLevels;
    viewCI.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(ctx.device, &viewCI, nullptr, &tex.imageView));

    tex.descriptorSet = rp.allocateTextureDescriptorSet(ctx, tex.imageView);
    return up;
}

Texture Scene::createTexture(VkContext& ctx, RenderPass& rp,
                             const unsigned char* pixels, int w, int h, int channels,
                             bool linear) {
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    TextureUpload up = recordTextureUpload(ctx, rp, cmd, pixels, w, h, channels, linear);
    ctx.endSingleTimeCommands(cmd);
    vmaDestroyBuffer(ctx.allocator, up.staging, up.stagingAlloc);
    return up.tex;
}

Texture Scene::createDummyWhiteTexture(VkContext& ctx, RenderPass& rp) {
    unsigned char white[] = {255, 255, 255, 255};
    return createTexture(ctx, rp, white, 1, 1, 4);
}

// ---------------------------------------------------------------------------
// glTF loading
// ---------------------------------------------------------------------------

static glm::mat4 nodeTransformToMat4(const cgltf_node* node) {
    glm::mat4 m(1.0f);
    if (node->has_matrix) {
        memcpy(&m, node->matrix, sizeof(float) * 16);
    } else {
        glm::mat4 T(1.0f), R(1.0f), S(1.0f);
        if (node->has_translation) {
            T[3][0] = node->translation[0];
            T[3][1] = node->translation[1];
            T[3][2] = node->translation[2];
        }
        if (node->has_rotation) {
            float qx = node->rotation[0], qy = node->rotation[1];
            float qz = node->rotation[2], qw = node->rotation[3];
            R[0][0] = 1 - 2*(qy*qy + qz*qz); R[0][1] = 2*(qx*qy + qz*qw); R[0][2] = 2*(qx*qz - qy*qw);
            R[1][0] = 2*(qx*qy - qz*qw); R[1][1] = 1 - 2*(qx*qx + qz*qz); R[1][2] = 2*(qy*qz + qx*qw);
            R[2][0] = 2*(qx*qz + qy*qw); R[2][1] = 2*(qy*qz - qx*qw); R[2][2] = 1 - 2*(qx*qx + qy*qy);
        }
        if (node->has_scale) {
            S[0][0] = node->scale[0];
            S[1][1] = node->scale[1];
            S[2][2] = node->scale[2];
        }
        m = T * R * S;
    }
    return m;
}

static glm::mat4 getWorldTransform(const cgltf_node* node) {
    glm::mat4 m = nodeTransformToMat4(node);
    if (node->parent) {
        m = getWorldTransform(node->parent) * m;
    }
    return m;
}

void Scene::registerMesh(Mesh& mesh) {
    mesh.meshIndex = (int)allMeshes_.size();
    allMeshes_.push_back(mesh);
}

void Scene::loadNumeralGLB(VkContext& ctx, const std::string& path,
                           std::vector<Mesh>& outMeshes) {
    cgltf_options options = {};
    cgltf_data* data = nullptr;
    cgltf_result result = cgltf_parse_file(&options, path.c_str(), &data);
    if (result != cgltf_result_success) {
        fprintf(stderr, "Failed to parse glTF: %s\n", path.c_str());
        abort();
    }
    result = cgltf_load_buffers(&options, data, path.c_str());
    if (result != cgltf_result_success) {
        fprintf(stderr, "Failed to load glTF buffers: %s\n", path.c_str());
        cgltf_free(data);
        abort();
    }

    for (cgltf_size ni = 0; ni < data->nodes_count; ++ni) {
        cgltf_node* node = &data->nodes[ni];
        if (!node->mesh) continue;

        // Numerals are positioned around the ring by their node transform;
        // the draw splits it into rotation and ring position.
        glm::mat4 worldXform = getWorldTransform(node);

        for (cgltf_size pi = 0; pi < node->mesh->primitives_count; ++pi) {
            cgltf_primitive* prim = &node->mesh->primitives[pi];
            if (prim->type != cgltf_primitive_type_triangles) continue;

            cgltf_accessor* posAcc = nullptr;
            cgltf_accessor* normAcc = nullptr;
            for (cgltf_size ai = 0; ai < prim->attributes_count; ++ai) {
                if (prim->attributes[ai].type == cgltf_attribute_type_position)
                    posAcc = prim->attributes[ai].data;
                else if (prim->attributes[ai].type == cgltf_attribute_type_normal)
                    normAcc = prim->attributes[ai].data;
            }
            if (!posAcc) continue;

            size_t vertexCount = posAcc->count;

            // The pipeline's interleaved layout: pos(3) normal(3) uv(2)
            // tangent(4). UV and tangent stay zero; the numeral draw samples
            // no texture and no normal map.
            std::vector<float> vertices(vertexCount * kFloatsPerVertex, 0.0f);
            for (size_t v = 0; v < vertexCount; ++v) {
                float* vert = &vertices[v * kFloatsPerVertex];
                cgltf_accessor_read_float(posAcc, v, vert + 0, 3);
                if (normAcc)
                    cgltf_accessor_read_float(normAcc, v, vert + 3, 3);
                else
                    vert[5] = 1.0f;  // +Z, facing out of the board
            }

            std::vector<uint32_t> indices;
            if (prim->indices) {
                indices.resize(prim->indices->count);
                for (size_t i = 0; i < prim->indices->count; ++i) {
                    indices[i] = (uint32_t)cgltf_accessor_read_index(prim->indices, i);
                }
            } else {
                indices.resize(vertexCount);
                for (size_t i = 0; i < vertexCount; ++i) indices[i] = (uint32_t)i;
            }

            Mesh mesh;
            mesh.name = node->name ? node->name : "";
            mesh.baseTransform = worldXform;
            mesh.indexCount = (uint32_t)indices.size();
            mesh.vertexCount = (uint32_t)vertexCount;

            uploadBuffer(ctx, mesh.vertexBuffer, mesh.vertexAlloc,
                         vertices.data(), vertices.size() * sizeof(float),
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            uploadBuffer(ctx, mesh.indexBuffer, mesh.indexAlloc,
                         indices.data(), indices.size() * sizeof(uint32_t),
                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

            outMeshes.push_back(mesh);
        }
    }

    cgltf_free(data);
}

// ---------------------------------------------------------------------------
// High-level scene loading
// ---------------------------------------------------------------------------

void Scene::loadAssets(VkContext& ctx, RenderPass& rp, const std::string& assetDir) {
    // Create dummy white texture first
    textures.push_back(createDummyWhiteTexture(ctx, rp));
    dummyWhiteTextureIndex = 0;

    // Create dummy flat normal map (pointing straight up in tangent space)
    {
        unsigned char flatNormal[] = {128, 128, 255, 255};
        textures.push_back(createTexture(ctx, rp, flatNormal, 1, 1, 4, true));
        dummyFlatNormalTextureIndex = (int)textures.size() - 1;
    }

    // Neutral environment probe for background-less frames. sRGB 128 decodes to
    // ~0.216 linear, which is about the radiance an ordinary room reflects, so
    // the ambient scale means the same thing whether or not a photo is bound.
    {
        unsigned char midGrey[] = {128, 128, 128, 255};
        textures.push_back(createTexture(ctx, rp, midGrey, 1, 1, 4));
        dummyEnvTextureIndex = (int)textures.size() - 1;
    }

    // Every scene dimension is a constant (constants.h). The board, the wire
    // frame and the darts are generated from those constants, and the
    // annotations and scoring read the same ones, so there is a single
    // definition of where each ring is.
    boardZ = BOARD_SURFACE_Z;
    ringRadiiBU = {
        {"inner_bull_r",   RING_RADII_BU.inner_bull_r},
        {"outer_bull_r",   RING_RADII_BU.outer_bull_r},
        {"triple_inner_r", RING_RADII_BU.triple_inner_r},
        {"triple_outer_r", RING_RADII_BU.triple_outer_r},
        {"triple_center_r", RING_RADII_BU.triple_center_r},
        {"double_inner_r", RING_RADII_BU.double_inner_r},
        {"double_outer_r", RING_RADII_BU.double_outer_r},
        {"board_r",        RING_RADII_BU.board_r},
    };

    // Load number variants.
    //
    // Bounded by NUM_FONT_VARIANTS, and every variant is required. The
    // randomizer draws a variant index in [0, NUM_FONT_VARIANTS) and the draw
    // call guards it against numberVariants.size(), so a loader that stopped
    // short would not fail -- the numerals would silently go missing on those
    // frames.
    for (int v = 0; v < NUM_FONT_VARIANTS; ++v) {
        std::string numPath = assetDir + "/numbers_variant_" + std::to_string(v) + ".glb";
        if (!fs::exists(numPath)) {
            fprintf(stderr, "Missing %s. NUM_FONT_VARIANTS is %d and every "
                            "variant must be present, or 1 in %d frames would "
                            "render with no numerals.\n",
                    numPath.c_str(), NUM_FONT_VARIANTS, NUM_FONT_VARIANTS);
            abort();
        }
        std::vector<Mesh> numMeshes;
        loadNumeralGLB(ctx, numPath, numMeshes);

        // Sort by name to ensure consistent order (Num_1 .. Num_20)
        std::sort(numMeshes.begin(), numMeshes.end(),
                  [](const Mesh& a, const Mesh& b) { return a.name < b.name; });
        for (Mesh& m : numMeshes) registerMesh(m);
        numberVariants.push_back(std::move(numMeshes));
    }

    // Printed-mark decal atlas: kDecalAtlasFontRows rows of kDecalAtlasGlyphCols
    // glyphs. Required. The randomizer places marks on every board whether or
    // not an atlas is bound, and the fallback white texture has alpha 1
    // everywhere, so a missing atlas does not give an unmarked board -- it
    // paints solid blocks and pills on the ring.
    {
        std::string decalPath = assetDir + "/decal_atlas.png";
        int w = 0, h = 0, ch = 0;
        unsigned char* px = stbi_load(decalPath.c_str(), &w, &h, &ch, 4);
        if (!px) {
            fprintf(stderr, "Failed to load %s: %s\n", decalPath.c_str(),
                    stbi_failure_reason());
            abort();
        }
        if (w % kDecalAtlasGlyphCols != 0 || h % kDecalAtlasFontRows != 0 ||
            w / kDecalAtlasGlyphCols != h / kDecalAtlasFontRows) {
            fprintf(stderr, "%s is %dx%d, which is not a grid of %d x %d "
                            "square cells\n", decalPath.c_str(), w, h,
                    kDecalAtlasGlyphCols, kDecalAtlasFontRows);
            abort();
        }
        decalAtlasTextureIndex = (int)textures.size();
        textures.push_back(createTexture(ctx, rp, px, w, h, 4));
        stbi_image_free(px);
        printf("  Decal atlas: %dx%d (%d glyphs x %d fonts)\n", w, h,
               kDecalAtlasGlyphCols, kDecalAtlasFontRows);
    }

    buildBoard(ctx, rp);
    buildSpider(ctx, rp);
    buildDarts(ctx, rp, 0x5EEDu);

    printf("Scene loaded: %zu textures, %zu dart variants, %zu number "
           "variants\n", textures.size(), dartVariants.size(),
           numberVariants.size());
}

void Scene::buildBoard(VkContext& ctx, RenderPass& rp) {
    BoardMeshParams params;
    params.radiusBU = (float)RING_RADII_BU.board_r;
    params.faceZ = (float)boardZ;

    std::vector<float> verts;
    std::vector<uint32_t> idx;
    buildBoardGeometry(params, verts, idx);

    boardFace = Mesh{};
    boardFace.name = "BoardFace";
    boardFace.indexCount = (uint32_t)idx.size();
    boardFace.vertexCount = (uint32_t)(verts.size() / kFloatsPerVertex);
    boardFace.baseTransform = glm::mat4(1.0f);

    uploadBuffer(ctx, boardFace.vertexBuffer, boardFace.vertexAlloc,
                 verts.data(), verts.size() * sizeof(float),
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    uploadBuffer(ctx, boardFace.indexBuffer, boardFace.indexAlloc,
                 idx.data(), idx.size() * sizeof(uint32_t),
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    registerMesh(boardFace);

    printf("  Board: generated %u tris at %d segments\n",
           boardFace.indexCount / 3, params.segments);
}

void Scene::buildSpider(VkContext& ctx, RenderPass& rp) {
    // The wire frame is generated rather than authored: it is exactly specified
    // geometry, and generating it keeps the wires on the same radii the
    // annotations and the scoring use. See wire_geometry.h.
    if (ringRadiiBU.empty()) {
        fprintf(stderr, "Warning: no ring radii, skipping spider\n");
        return;
    }

    // One mesh per geometry variant. Thickness and profile are vertex data, so
    // unlike the wire's colour they cannot be randomised per frame -- the only
    // way to vary them is to build several and choose at draw time.
    for (int v = 0; v < kSpiderVariants; ++v) {
        const WireParams params = spiderVariant(v);

        std::vector<float> verts;
        std::vector<uint32_t> idx;
        buildSpiderGeometry(ringRadiiBU, boardZ, params, verts, idx);
        if (idx.empty()) continue;

        Mesh spider;
        spider.name = "Spider" + std::to_string(v);
        spider.indexCount = (uint32_t)idx.size();
        spider.vertexCount = (uint32_t)(verts.size() / kFloatsPerVertex);
        // No material here: the frame's randomised wire colour, metallic and
        // roughness are supplied at draw time, because a board may have
        // bright steel, dulled steel or black-coated wire and the difference
        // is far more visible than the profile is.
        spider.baseTransform = glm::mat4(1.0f);

        uploadBuffer(ctx, spider.vertexBuffer, spider.vertexAlloc,
                     verts.data(), verts.size() * sizeof(float),
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        uploadBuffer(ctx, spider.indexBuffer, spider.indexAlloc,
                     idx.data(), idx.size() * sizeof(uint32_t),
                     VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        registerMesh(spider);
        spiderVariants.push_back(spider);

        // Half-extent above the wire's centre line: the radius for round
        // wire, the blade's half-height for blade. The crown therefore sits at
        // standoff + halfExtent above the face, and standoff - halfExtent is
        // the gap underneath -- a real spider is stapled on top of the sisal,
        // so that gap must stay positive or the wire beds into the face.
        const float halfExtent = params.thickness * 0.5f
                               * (params.blade ? params.bladeHeightRatio : 1.0f);
        printf("  Spider %d: %u verts, %u tris (%s %.2fmm, crown %.2fmm proud, "
               "%.2fmm gap)\n",
               v, spider.vertexCount, spider.indexCount / 3,
               params.blade ? "blade" : "round", params.thickness * 100.0f,
               (params.standoff + halfExtent) * 100.0f,
               (params.standoff - halfExtent) * 100.0f);
    }
}

void Scene::buildVariantMeshes(VkContext& ctx, VkCommandBuffer cmd,
                               const DartBuild& build, int variant,
                               std::vector<Mesh>& out,
                               std::vector<PendingStage>& pending) {
    struct PartSrc { const DartPartMesh* m; const char* tag; DartPart part; };
    const PartSrc parts[3] = {
        {&build.point,  "Point",  DartPart::Point},
        {&build.barrel, "Barrel", DartPart::Metal},
        {&build.tail,   "Tail",   DartPart::Flight},
    };
    out.clear();
    for (const PartSrc& src : parts) {
        if (src.m->indices.empty()) continue;
        Mesh mesh;
        mesh.name = "Dart" + std::to_string(variant) + src.tag;
        mesh.dartPart = src.part;
        mesh.indexCount = (uint32_t)src.m->indices.size();
        mesh.vertexCount = (uint32_t)(src.m->vertices.size() / kFloatsPerVertex);
        // Material comes from the frame's DartMaterial at draw time, which is
        // filled from this variant's finish, so a brass barrel stays
        // brass-coloured as well as brass-sized.
        mesh.baseTransform = glm::mat4(1.0f);

        stageBuffer(ctx, cmd, mesh.vertexBuffer, mesh.vertexAlloc,
                    src.m->vertices.data(),
                    src.m->vertices.size() * sizeof(float),
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, pending);
        stageBuffer(ctx, cmd, mesh.indexBuffer, mesh.indexAlloc,
                    src.m->indices.data(),
                    src.m->indices.size() * sizeof(uint32_t),
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT, pending);
        out.push_back(std::move(mesh));
    }
}

bool Scene::refreshDartVariant(VkContext& ctx, int variant, uint32_t seed,
                               std::vector<size_t>& outMeshIndices) {
    outMeshIndices.clear();
    if (variant < 0 || variant >= (int)dartVariants.size()) return false;
    DartVariant& slot = dartVariants[variant];
    // A variant whose primitive count would change cannot be swapped in place:
    // allMeshes_ slots are fixed and blasList mirrors them by index. Every
    // design has the same three parts, so this only guards a future one that
    // does not.
    if (slot.meshIndices.size() != slot.meshes.size()) return false;

    std::mt19937 rng(seed);
    DartTessellation tess;
    DartBuild build;
    buildDart(sampleDart(rng), tess, build);

    // One command buffer for all six copies, one submit, one wait.
    std::vector<PendingStage> pending;
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    std::vector<Mesh> fresh;
    buildVariantMeshes(ctx, cmd, build, variant, fresh, pending);
    ctx.endSingleTimeCommands(cmd);
    for (PendingStage& ps : pending)
        vmaDestroyBuffer(ctx.allocator, ps.buffer, ps.alloc);
    if (fresh.size() != slot.meshes.size()) {
        for (Mesh& m : fresh) {
            if (m.vertexBuffer) vmaDestroyBuffer(ctx.allocator, m.vertexBuffer, m.vertexAlloc);
            if (m.indexBuffer)  vmaDestroyBuffer(ctx.allocator, m.indexBuffer, m.indexAlloc);
        }
        return false;
    }

    for (size_t k = 0; k < slot.meshes.size(); ++k) {
        Mesh& old = allMeshes_[slot.meshIndices[k]];
        if (old.vertexBuffer)
            vmaDestroyBuffer(ctx.allocator, old.vertexBuffer, old.vertexAlloc);
        if (old.indexBuffer)
            vmaDestroyBuffer(ctx.allocator, old.indexBuffer, old.indexAlloc);
        fresh[k].meshIndex = (int)slot.meshIndices[k];
        old = fresh[k];
        slot.meshes[k] = fresh[k];
        outMeshIndices.push_back(slot.meshIndices[k]);
    }
    extractGeometry(build, slot.geom);
    dartGeometries[variant] = slot.geom;
    return true;
}

void Scene::buildDarts(VkContext& ctx, RenderPass& rp, uint32_t seed) {
    // A pool, for the same reason the spider is a pool: the geometry is vertex
    // data and cannot vary per frame, so the only way to vary it is to build
    // several and choose at draw time. Unlike the spider, the choice is per
    // TURN and not per frame, and all three darts take the same one -- a
    // player throws a matched set.
    std::mt19937 rng(seed);
    DartTessellation tess;
    DartBuild build;

    size_t totalTris = 0, totalVerts = 0;
    for (int v = 0; v < kDartVariants; ++v) {
        const DartParams params = sampleDart(rng);
        buildDart(params, tess, build);

        DartVariant variant;
        extractGeometry(build, variant.geom);

        std::vector<Mesh> fresh;
        std::vector<PendingStage> pending;
        VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
        buildVariantMeshes(ctx, cmd, build, v, fresh, pending);
        ctx.endSingleTimeCommands(cmd);
        for (PendingStage& ps : pending)
            vmaDestroyBuffer(ctx.allocator, ps.buffer, ps.alloc);
        for (Mesh& mesh : fresh) {
            totalTris += mesh.indexCount / 3;
            totalVerts += mesh.vertexCount;
            registerMesh(mesh);
            variant.meshIndices.push_back((size_t)mesh.meshIndex);
            variant.meshes.push_back(std::move(mesh));
        }

        dartGeometries.push_back(variant.geom);
        dartVariants.push_back(std::move(variant));

        if (v < 3)
            printf("  Dart %d: barrel %.1fmm dia %.2f (%s), total %.1fmm, "
                   "%zu verts\n", v, build.barrelLengthMm, build.barrelDiaMm,
                   params.barrel.material == kTungsten ? "tungsten"
                   : params.barrel.material == kBrass ? "brass" : "nickel",
                   build.totalLengthMm,
                   (build.point.vertices.size() + build.barrel.vertices.size()
                    + build.tail.vertices.size()) / kFloatsPerVertex);
    }
    printf("  Darts: %d generated variants, %zu verts / %zu tris total\n",
           kDartVariants, totalVerts, totalTris);
}

void Scene::destroy(VkContext& ctx) {
    // Free all mesh GPU resources from the master list (single owner)
    for (auto& m : allMeshes_) {
        if (m.vertexBuffer) vmaDestroyBuffer(ctx.allocator, m.vertexBuffer, m.vertexAlloc);
        if (m.indexBuffer)  vmaDestroyBuffer(ctx.allocator, m.indexBuffer, m.indexAlloc);
    }

    for (auto& tex : textures) {
        vkDestroyImageView(ctx.device, tex.imageView, nullptr);
        vmaDestroyImage(ctx.allocator, tex.image, tex.alloc);
    }
}

} // namespace dart
