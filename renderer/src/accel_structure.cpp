#include "accel_structure.h"

#include "constants.h"

#include <chrono>
#include "render_pass.h"
#include "board_transforms.h"

#include <glm/gtc/matrix_transform.hpp>
#include <cstdio>
#include <cstring>

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

static VkTransformMatrixKHR toVkTransform(const glm::mat4& m) {
    VkTransformMatrixKHR x{};
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 4; col++)
            x.matrix[row][col] = m[col][row];  // GLM col-major → VK row-major
    return x;
}

AccelBuffer AccelStructure::createBuffer(VkContext& ctx, VkDeviceSize size,
                                          VkBufferUsageFlags usage, bool hostVisible) {
    AccelBuffer ab;

    VkBufferCreateInfo bufCI = {};
    bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size = size;
    bufCI.usage = usage;

    VmaAllocationCreateInfo allocCI = {};
    if (hostVisible) {
        allocCI.usage = VMA_MEMORY_USAGE_AUTO;
        allocCI.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT;
    } else {
        allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    }

    VmaAllocationInfo allocInfo;
    VK_CHECK(vmaCreateBuffer(ctx.allocator, &bufCI, &allocCI,
                             &ab.buffer, &ab.alloc, &allocInfo));

    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo addrInfo = {};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = ab.buffer;
        ab.deviceAddress = ctx.vkGetBufferDeviceAddressKHR(ctx.device, &addrInfo);
    }

    return ab;
}

void AccelStructure::buildBLAS(VkContext& ctx, Scene& scene) {
    std::vector<size_t> all(scene.allMeshes_.size());
    for (size_t i = 0; i < all.size(); ++i) all[i] = i;
    buildBLASFor(ctx, scene, all);
}

void AccelStructure::refreshBLAS(VkContext& ctx, Scene& scene,
                                 const std::vector<size_t>& meshIndices) {
    // The caller has already waited for the device, because the buffers these
    // structures were built over have been freed and replaced.
    for (size_t i : meshIndices) {
        if (i >= blasList.size()) continue;
        BLAS& b = blasList[i];
        if (b.handle)
            ctx.vkDestroyAccelerationStructureKHR(ctx.device, b.handle, nullptr);
        if (b.buffer.buffer)
            vmaDestroyBuffer(ctx.allocator, b.buffer.buffer, b.buffer.alloc);
        b = BLAS{};
    }
    buildBLASFor(ctx, scene, meshIndices);
}

/// Build (or rebuild) the acceleration structures for `indices`.
///
/// Shared by buildBLAS and the rolling refresh, which rebuilds three meshes
/// without touching the other few hundred. The scratch buffer is sized to the
/// largest of the set and shared across it.
void AccelStructure::buildBLASFor(VkContext& ctx, Scene& scene,
                                  const std::vector<size_t>& indices) {
    // Filter out empty/invalid meshes
    std::vector<size_t> validIndices;
    for (size_t i : indices) {
        if (i >= scene.allMeshes_.size()) continue;
        auto& m = scene.allMeshes_[i];
        if (m.indexCount > 0 && m.vertexCount > 0 &&
            m.vertexBuffer != VK_NULL_HANDLE && m.indexBuffer != VK_NULL_HANDLE)
            validIndices.push_back(i);
    }

    if (blasList.size() < scene.allMeshes_.size())
        blasList.resize(scene.allMeshes_.size());

    VkDeviceSize maxScratchSize = 0;

    // First pass: get build sizes
    struct BLASBuildInfo {
        VkAccelerationStructureGeometryKHR geometry;
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo;
        VkAccelerationStructureBuildSizesInfoKHR sizesInfo;
        uint32_t primitiveCount;
    };
    std::vector<BLASBuildInfo> buildInfos(scene.allMeshes_.size());

    for (size_t vi = 0; vi < validIndices.size(); ++vi) {
        size_t i = validIndices[vi];
        auto& mesh = scene.allMeshes_[i];
        auto& info = buildInfos[i];

        // Get device addresses of vertex/index buffers
        VkBufferDeviceAddressInfo vbAddr = {};
        vbAddr.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        vbAddr.buffer = mesh.vertexBuffer;
        VkDeviceAddress vertexAddress = ctx.vkGetBufferDeviceAddressKHR(ctx.device, &vbAddr);

        VkBufferDeviceAddressInfo ibAddr = {};
        ibAddr.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        ibAddr.buffer = mesh.indexBuffer;
        VkDeviceAddress indexAddress = ctx.vkGetBufferDeviceAddressKHR(ctx.device, &ibAddr);

        info.geometry = {};
        info.geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        info.geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        // Flights are built non-opaque so shadow rays generate candidate hits
        // for them instead of being terminated. The shader then accumulates
        // tinted transmission rather than treating the flight as a blocker.
        info.geometry.flags = (mesh.dartPart == DartPart::Flight)
                                ? 0u
                                : VK_GEOMETRY_OPAQUE_BIT_KHR;

        auto& triangles = info.geometry.geometry.triangles;
        triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = vertexAddress;
        triangles.vertexStride = sizeof(float) * 12;  // pos(3) + normal(3) + uv(2) + tangent(4)
        triangles.maxVertex = mesh.vertexCount - 1;
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = indexAddress;

        info.primitiveCount = mesh.indexCount / 3;

        info.buildInfo = {};
        info.buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        info.buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        info.buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        info.buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        info.buildInfo.geometryCount = 1;
        info.buildInfo.pGeometries = &info.geometry;

        info.sizesInfo = {};
        info.sizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        ctx.vkGetAccelerationStructureBuildSizesKHR(
            ctx.device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &info.buildInfo,
            &info.primitiveCount,
            &info.sizesInfo);

        if (info.sizesInfo.buildScratchSize > maxScratchSize)
            maxScratchSize = info.sizesInfo.buildScratchSize;
    }

    // Create shared scratch buffer
    if (maxScratchSize == 0) maxScratchSize = 256;
    AccelBuffer scratch = createBuffer(ctx, maxScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        false);

    // Second pass: create the structures, then build them in BATCHES.
    //
    // One submit per structure means one vkQueueWaitIdle per structure. That
    // is invisible at startup but not in a rolling refresh, where the round
    // trips would dominate ~1.5MB of copies and 40k triangles of actual work.
    //
    // Builds within a batch share the scratch buffer, so they are serialised
    // with a barrier between them rather than run concurrently. That is still
    // one round trip instead of N.
    constexpr size_t kBatch = 32;
    for (size_t base = 0; base < validIndices.size(); base += kBatch) {
        const size_t end = std::min(base + kBatch, validIndices.size());
        for (size_t vi = base; vi < end; ++vi) {
            size_t i = validIndices[vi];
            auto& info = buildInfos[i];
            auto& blas = blasList[i];

            blas.buffer = createBuffer(ctx, info.sizesInfo.accelerationStructureSize,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                false);

            VkAccelerationStructureCreateInfoKHR asCI = {};
            asCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            asCI.buffer = blas.buffer.buffer;
            asCI.size = info.sizesInfo.accelerationStructureSize;
            asCI.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            VK_CHECK(ctx.vkCreateAccelerationStructureKHR(ctx.device, &asCI,
                                                          nullptr, &blas.handle));

            info.buildInfo.dstAccelerationStructure = blas.handle;
            info.buildInfo.scratchData.deviceAddress = scratch.deviceAddress;
        }

        VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
        for (size_t vi = base; vi < end; ++vi) {
            const size_t i = validIndices[vi];
            VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {};
            rangeInfo.primitiveCount = buildInfos[i].primitiveCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;
            ctx.vkCmdBuildAccelerationStructuresKHR(cmd, 1,
                                                    &buildInfos[i].buildInfo,
                                                    &pRange);
            // The next build writes the same scratch, so it must not start
            // until this one has finished reading it.
            if (vi + 1 < end) {
                VkMemoryBarrier mb = {};
                mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
                                 | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                vkCmdPipelineBarrier(
                    cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                    0, 1, &mb, 0, nullptr, 0, nullptr);
            }
        }
        ctx.endSingleTimeCommands(cmd);
    }

    // Free scratch
    vmaDestroyBuffer(ctx.allocator, scratch.buffer, scratch.alloc);
}

void AccelStructure::createTLAS(VkContext& ctx, Scene& scene,
                                 VkDescriptorPool pool, VkDescriptorSetLayout tlasLayout) {
    // Count max instances: 1 board + spider + 20 numbers
    //                      + 3 darts * max sub-meshes
    maxInstances = 1;  // board
    maxInstances += 1;  // spider
    if (!scene.numberVariants.empty())
        maxInstances += (uint32_t)scene.numberVariants[0].size();
    // Three darts, all of one variant, so the cap is three times the largest
    // variant's primitive count rather than the sum over a fixed three groups.
    {
        size_t widest = 0;
        for (const auto& v : scene.dartVariants)
            widest = std::max(widest, v.meshes.size());
        maxInstances += (uint32_t)(3 * widest);
    }

    // Get TLAS build sizes (same for all frames)
    VkAccelerationStructureGeometryKHR tlasGeo = {};
    tlasGeo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlasGeo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    // Deliberately not VK_GEOMETRY_OPAQUE_BIT_KHR.
    //
    // Setting it here marks the whole instances geometry opaque, which
    // overrides the per-mesh flags the BLAS was built with: no candidate hits
    // are ever surfaced, rayQueryProceedEXT has nothing to yield, and the
    // transmission walk in shadowTransmission (surface_rt.frag) never runs.
    // Nothing looks broken when that happens -- flights just cast plain
    // opaque shadows.
    //
    // Opacity for instances belongs in VkGeometryInstanceFlagBitsKHR, per
    // instance, which is where it is set.
    tlasGeo.flags = 0;
    tlasGeo.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeo.geometry.instances.arrayOfPointers = VK_FALSE;
    // deviceAddress will be set per-frame, but sizes query doesn't need it
    tlasGeo.geometry.instances.data.deviceAddress = 0;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    // Rebuilt from scratch every frame, never refit: each frame is an
    // independent random scene, so there is no previous structure worth
    // updating, and ALLOW_UPDATE would cost memory and trace speed.
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &tlasGeo;

    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {};
    sizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    ctx.vkGetAccelerationStructureBuildSizesKHR(
        ctx.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        &maxInstances,
        &sizesInfo);

    VkDeviceSize scratchSize = sizesInfo.buildScratchSize;
    if (scratchSize == 0) scratchSize = 256;
    VkDeviceSize instanceSize = maxInstances * sizeof(VkAccelerationStructureInstanceKHR);

    // Create per-frame TLAS resources
    for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        auto& tf = tlasFrames[i];

        // Instance buffer (host-visible, persistently mapped)
        tf.instanceBuffer = createBuffer(ctx, instanceSize,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            true);

        // TLAS buffer
        tf.tlasBuffer = createBuffer(ctx, sizesInfo.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            false);

        // Create TLAS
        VkAccelerationStructureCreateInfoKHR asCI = {};
        asCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        asCI.buffer = tf.tlasBuffer.buffer;
        asCI.size = sizesInfo.accelerationStructureSize;
        asCI.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        VK_CHECK(ctx.vkCreateAccelerationStructureKHR(ctx.device, &asCI, nullptr, &tf.tlas));

        // Scratch buffer
        tf.scratchBuffer = createBuffer(ctx, scratchSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            false);

        // Material buffer: host-visible and persistently mapped, like the
        // instance buffer it parallels. Written in the same loop, so an index
        // can never mean one thing to the TLAS and another to the shader.
        tf.materialBuffer = createBuffer(ctx, maxInstances * sizeof(GpuMaterial),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);

        // Allocate and write TLAS descriptor set
        VkDescriptorSetAllocateInfo dsAllocInfo = {};
        dsAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsAllocInfo.descriptorPool = pool;
        dsAllocInfo.descriptorSetCount = 1;
        dsAllocInfo.pSetLayouts = &tlasLayout;
        VK_CHECK(vkAllocateDescriptorSets(ctx.device, &dsAllocInfo, &tf.tlasDescSet));

        VkWriteDescriptorSetAccelerationStructureKHR asWrite = {};
        asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        asWrite.accelerationStructureCount = 1;
        asWrite.pAccelerationStructures = &tf.tlas;

        VkDescriptorBufferInfo matInfo = {};
        matInfo.buffer = tf.materialBuffer.buffer;
        matInfo.offset = 0;
        matInfo.range  = maxInstances * sizeof(GpuMaterial);

        VkWriteDescriptorSet writes[2] = {};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = tf.tlasDescSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        writes[0].pNext = &asWrite;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = tf.tlasDescSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &matInfo;

        vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);
    }
}

void AccelStructure::init(VkContext& ctx, Scene& scene,
                           VkDescriptorPool pool, VkDescriptorSetLayout tlasLayout) {
    buildBLAS(ctx, scene);
    createTLAS(ctx, scene, pool, tlasLayout);
}

void AccelStructure::updateTLAS(VkContext& ctx, VkCommandBuffer cmd,
                                 Scene& scene, const FrameState& state,
                                 int frameIndex) {
    auto& tf = tlasFrames[frameIndex];

    // Get mapped pointer to instance buffer
    VmaAllocationInfo allocInfo;
    vmaGetAllocationInfo(ctx.allocator, tf.instanceBuffer.alloc, &allocInfo);
    auto* instances = static_cast<VkAccelerationStructureInstanceKHR*>(allocInfo.pMappedData);

    VmaAllocationInfo matAllocInfo;
    vmaGetAllocationInfo(ctx.allocator, tf.materialBuffer.alloc, &matAllocInfo);
    auto* materials = static_cast<GpuMaterial*>(matAllocInfo.pMappedData);
    uint32_t instanceCount = 0;

    // customIndex is the instance's own index, which is also its slot in the
    // material buffer, so a shadow or reflection ray can look up what it hit.
    // Materials are written here rather than anywhere else precisely so the
    // index cannot mean two different things.
    auto addInstance = [&](const Mesh& mesh, const glm::mat4& transform,
                           const GpuMaterial& material = GpuMaterial{}) {
        if (instanceCount >= maxInstances) return;
        // blasList mirrors allMeshes_, so a mesh's slot there is its BLAS.
        const int blasIdx = mesh.meshIndex;
        if (blasIdx < 0 || blasIdx >= (int)blasList.size()) return;
        // Skip meshes with no BLAS (empty/invalid meshes)
        if (blasList[blasIdx].handle == VK_NULL_HANDLE) return;

        materials[instanceCount] = material;

        auto& inst = instances[instanceCount];
        inst.transform = toVkTransform(transform);
        inst.instanceCustomIndex = instanceCount;
        inst.mask = 0xFF;
        inst.instanceShaderBindingTableRecordOffset = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        // A translucent flight must yield candidate hits so the shadow ray can
        // accumulate its tint and carry on past it. Stated per instance rather
        // than left to the BLAS flag: whether a flight is translucent is a
        // per-frame material property, while the BLAS is built once and shared
        // by every dart.
        if (material.albedo.a > 1.5 &&
            (material.transmit.r + material.transmit.g + material.transmit.b) > 0.0f)
            inst.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
        else
            inst.flags |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;

        // Get BLAS device address
        VkAccelerationStructureDeviceAddressInfoKHR addrInfo = {};
        addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addrInfo.accelerationStructure = blasList[blasIdx].handle;
        inst.accelerationStructureReference =
            ctx.vkGetAccelerationStructureDeviceAddressKHR(ctx.device, &addrInfo);

        instanceCount++;
    };

    // Kinds must match the GpuMaterial documentation in render_pass.h.
    auto flat = [](const glm::vec3& c) {
        GpuMaterial m; m.albedo = glm::vec4(c, 0.0f); return m;
    };

    // Replicate the draw order from recordFrame()
    glm::mat4 boardRot = glm::rotate(glm::mat4(1.0f), state.boardRotation, glm::vec3(0, 0, 1));

    glm::mat4 faceRot = glm::rotate(glm::mat4(1.0f), state.boardFaceRotation,
                                    glm::vec3(0, 0, 1));

    // 1. Board face. Takes boardFaceRot to match the raster pass. The face is a
    // flat disc, so rotating it about Z leaves the shadow-casting silhouette
    // unchanged -- but the two paths must agree (see board_transforms.h), and
    // this would matter the moment the face gains any relief.
    {
        // Kind 1: one instance covering twenty differently coloured beds, so a
        // flat albedo would be a lie. The shader evaluates boardFaceAlbedo at
        // the hit point instead -- that is what puts red and green on a barrel.
        GpuMaterial m; m.albedo = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
        addInstance(scene.boardFace, boardRot * faceRot * scene.boardFace.baseTransform, m);
    }

    // 1b. Spider. Included so the wires cast the contact shadows and occlude
    // ambient light the way they do on a real board -- that shading is most of
    // what makes them legible.
    if (const Mesh* spider = scene.spiderFor(state.spiderVariant)) {
        addInstance(*spider, boardRot * faceRot * spider->baseTransform,
                    flat(state.wireColor));
    }

    // 2. Number meshes
    if (state.fontVariant < (int)scene.numberVariants.size()) {
        auto& numbers = scene.numberVariants[state.fontVariant];
        glm::mat4 scaleM = glm::scale(glm::mat4(1.0f), glm::vec3(state.numberScale));
        for (auto& numMesh : numbers) {
            glm::vec3 pos = glm::vec3(numMesh.baseTransform[3]);
            glm::mat4 localXform = numMesh.baseTransform;
            localXform[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            // Must match the raster pass exactly, or the numerals cast shadows
            // in an orientation they are not drawn in.
            addInstance(numMesh, numeralModelMatrix(boardRot, pos, scaleM, localXform,
                                                 state.numberRingOffset),
                        flat(glm::vec3(state.numberMetallic ? 0.75f : 1.0f)));
        }
    }

    // 3. Darts
    const Scene::DartVariant* dv = scene.dartFor(state.dartVariant);
    for (int i = 0; dv != nullptr && i < state.numDarts; ++i) {
        const DartMaterial& dm = state.dartMaterials[i];
        for (const auto& dartMesh : dv->meshes) {
            GpuMaterial m;
            if (dartMesh.dartPart == DartPart::Flight) {
                // Kind 2: translucent. transmit is what survives passing
                // through, and is what tints the shadow the flight casts.
                m.albedo = glm::vec4(dm.flightColor, 2.0f);
                // What survives the flight is its own colour scaled by how
                // translucent it is; a fully opaque flight passes nothing.
                const float t = 1.0f - glm::clamp(dm.flightAlpha, 0.0f, 1.0f);
                m.transmit = glm::vec4(dm.flightColor * t, 0.0f);
            } else if (dartMesh.dartPart == DartPart::Point) {
                m.albedo = glm::vec4(dm.pointColor, 0.0f);
            } else {
                m.albedo = glm::vec4(dm.metalColor, 0.0f);
            }
            addInstance(dartMesh, state.dartTransforms[i], m);
        }
    }

    // Flush instance buffer writes
    vmaFlushAllocation(ctx.allocator, tf.instanceBuffer.alloc, 0, VK_WHOLE_SIZE);
    vmaFlushAllocation(ctx.allocator, tf.materialBuffer.alloc, 0, VK_WHOLE_SIZE);

    // Build TLAS
    VkAccelerationStructureGeometryKHR tlasGeo = {};
    tlasGeo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlasGeo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    // Deliberately not VK_GEOMETRY_OPAQUE_BIT_KHR.
    //
    // Setting it here marks the whole instances geometry opaque, which
    // overrides the per-mesh flags the BLAS was built with: no candidate hits
    // are ever surfaced, rayQueryProceedEXT has nothing to yield, and the
    // transmission walk in shadowTransmission (surface_rt.frag) never runs.
    // Nothing looks broken when that happens -- flights just cast plain
    // opaque shadows.
    //
    // Opacity for instances belongs in VkGeometryInstanceFlagBitsKHR, per
    // instance, which is where it is set.
    tlasGeo.flags = 0;
    tlasGeo.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeo.geometry.instances.arrayOfPointers = VK_FALSE;
    tlasGeo.geometry.instances.data.deviceAddress = tf.instanceBuffer.deviceAddress;

    VkAccelerationStructureBuildGeometryInfoKHR bldInfo = {};
    bldInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    bldInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    bldInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    bldInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bldInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    bldInfo.dstAccelerationStructure = tf.tlas;
    bldInfo.geometryCount = 1;
    bldInfo.pGeometries = &tlasGeo;
    bldInfo.scratchData.deviceAddress = tf.scratchBuffer.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {};
    rangeInfo.primitiveCount = instanceCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;

    ctx.vkCmdBuildAccelerationStructuresKHR(cmd, 1, &bldInfo, &pRange);

    // Memory barrier: TLAS build → fragment shader read
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void AccelStructure::destroy(VkContext& ctx) {
    for (int i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        auto& tf = tlasFrames[i];
        if (tf.tlas)
            ctx.vkDestroyAccelerationStructureKHR(ctx.device, tf.tlas, nullptr);
        if (tf.tlasBuffer.buffer)
            vmaDestroyBuffer(ctx.allocator, tf.tlasBuffer.buffer, tf.tlasBuffer.alloc);
        if (tf.instanceBuffer.buffer)
            vmaDestroyBuffer(ctx.allocator, tf.instanceBuffer.buffer, tf.instanceBuffer.alloc);
        if (tf.scratchBuffer.buffer)
            vmaDestroyBuffer(ctx.allocator, tf.scratchBuffer.buffer, tf.scratchBuffer.alloc);
        if (tf.materialBuffer.buffer)
            vmaDestroyBuffer(ctx.allocator, tf.materialBuffer.buffer, tf.materialBuffer.alloc);
    }
    for (auto& blas : blasList) {
        if (blas.handle)
            ctx.vkDestroyAccelerationStructureKHR(ctx.device, blas.handle, nullptr);
        if (blas.buffer.buffer)
            vmaDestroyBuffer(ctx.allocator, blas.buffer.buffer, blas.buffer.alloc);
    }
}

void rollDartVariant(VkContext& ctx, Scene& scene, AccelStructure& accel,
                     int frameId, int inFlightVariant) {
    if (scene.dartVariants.empty()) return;
    if (kDartRefreshFrames <= 0) return;
    if (frameId <= 0 || frameId % kDartRefreshFrames != 0) return;

    // Round robin, so every slot is replaced in turn rather than the same few
    // being hit repeatedly by chance.
    int slot = (frameId / kDartRefreshFrames) % (int)scene.dartVariants.size();
    if (slot == inFlightVariant)
        slot = (slot + 1) % (int)scene.dartVariants.size();

    const auto t0 = std::chrono::steady_clock::now();
    vkDeviceWaitIdle(ctx.device);
    const auto tWait = std::chrono::steady_clock::now();
    std::vector<size_t> changed;
    // Seed from the frame and the slot so a resumed run does not regenerate
    // the same replacements in the same order.
    const uint32_t seed = (uint32_t)(frameId * 2654435761u) ^ (uint32_t)(slot * 40503u);
    if (!scene.refreshDartVariant(ctx, slot, seed, changed) || changed.empty())
        return;
    // Only the acceleration structures are RT-specific. The raster path draws
    // from the same buffers, so the refresh itself has to happen either way --
    // gating the whole function on rtEnabled would leave the pool static
    // wherever ray tracing is unavailable.
    const auto tBuild = std::chrono::steady_clock::now();
    if (ctx.rtEnabled) accel.refreshBLAS(ctx, scene, changed);
    const auto tEnd = std::chrono::steady_clock::now();

    // Timing reported for the first few refreshes, so the cost is visible in
    // the log rather than assumed.
    static int announced = 0;
    if (announced < 3) {
        ++announced;
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        printf("  Darts: refreshed variant %d in %.1f ms "
               "(wait %.1f, build+upload %.1f, blas %.1f), every %d frames\n",
               slot, ms(t0, tEnd), ms(t0, tWait), ms(tWait, tBuild),
               ms(tBuild, tEnd), kDartRefreshFrames);
    }
}

} // namespace dart
