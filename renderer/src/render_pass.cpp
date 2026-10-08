#include "render_pass.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

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

VkShaderModule RenderPass::loadShaderModule(VkDevice device, const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open shader: %s\n", path.c_str());
        abort();
    }
    size_t fileSize = file.tellg();
    std::vector<uint32_t> code((fileSize + 3) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), fileSize);

    VkShaderModuleCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = fileSize;
    ci.pCode = code.data();

    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &mod));
    return mod;
}

// Abort unless the device can do everything the colour pipeline asks of
// kColorFormat. Every capability checked here is mandatory for
// R8G8B8A8_SRGB in the Vulkan spec, so a failure means a broken driver rather
// than an unusual one -- but a driver that silently skipped the sRGB
// conversion or the linear filter would change every rendered pixel without
// an error, so it is checked rather than assumed.
static void requireColorFormatSupport(VkContext& ctx, VkSampleCountFlagBits msaa) {
    VkFormatProperties fp = {};
    vkGetPhysicalDeviceFormatProperties(ctx.physicalDevice, kColorFormat, &fp);
    const VkFormatFeatureFlags need =
        VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
        VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
        VK_FORMAT_FEATURE_BLIT_SRC_BIT |
        VK_FORMAT_FEATURE_BLIT_DST_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    if ((fp.optimalTilingFeatures & need) != need) {
        fprintf(stderr, "R8G8B8A8_SRGB lacks required optimal-tiling features "
                        "(have 0x%x, need 0x%x)\n",
                fp.optimalTilingFeatures, need);
        abort();
    }

    VkImageFormatProperties ifp = {};
    VkResult r = vkGetPhysicalDeviceImageFormatProperties(
        ctx.physicalDevice, kColorFormat, VK_IMAGE_TYPE_2D,
        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 0, &ifp);
    if (r != VK_SUCCESS || !(ifp.sampleCounts & msaa)) {
        fprintf(stderr, "R8G8B8A8_SRGB colour attachments do not support %dx "
                        "MSAA on this device\n", (int)msaa);
        abort();
    }
}

void RenderPass::init(VkContext& ctx, uint32_t w, uint32_t h) {
    width = w;
    height = h;

    requireColorFormatSupport(ctx, msaaSamples);

    // --- Render pass (MSAA with resolve) ---
    // Attachment 0: MSAA color
    VkAttachmentDescription msaaColorAtt = {};
    msaaColorAtt.format = kColorFormat;
    msaaColorAtt.samples = msaaSamples;
    msaaColorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    msaaColorAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    msaaColorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    msaaColorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    msaaColorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    msaaColorAtt.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // Attachment 1: MSAA depth
    VkAttachmentDescription depthAtt = {};
    depthAtt.format = VK_FORMAT_D32_SFLOAT;
    depthAtt.samples = msaaSamples;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // Attachment 2: resolve target (1x, readback)
    VkAttachmentDescription resolveAtt = {};
    resolveAtt.format = kColorFormat;
    resolveAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    resolveAtt.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    resolveAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    resolveAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    resolveAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    resolveAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    resolveAtt.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkAttachmentReference colorRef = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkAttachmentReference resolveRef = {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pResolveAttachments = &resolveRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkAttachmentDescription attachments[] = {msaaColorAtt, depthAtt, resolveAtt};

    VkSubpassDependency deps[2] = {};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = 0;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    // The resolve target is blitted straight after the pass ends
    // (recordFrame), so its resolve writes must be available to the transfer
    // stage. The transition to TRANSFER_SRC_OPTIMAL (finalLayout) is ordered
    // inside this dependency.
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    VkRenderPassCreateInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 3;
    rpInfo.pAttachments = attachments;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 2;
    rpInfo.pDependencies = deps;

    VK_CHECK(vkCreateRenderPass(ctx.device, &rpInfo, nullptr, &renderPass));

    // --- Descriptor set layouts ---
    // Set 0: UBO
    {
        VkDescriptorSetLayoutBinding binding = {};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings = &binding;
        VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &descSetLayout0));
    }

    // Set 1: Texture sampler
    {
        VkDescriptorSetLayoutBinding binding = {};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings = &binding;
        VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &descSetLayout1));
    }

    // descSetLayout2: ray-tracing resources, bound at set 3 (RT only).
    //
    // The per-instance material table rides in this set rather than taking a
    // set of its own because the pipeline already binds six, and the Vulkan
    // floor for maxBoundDescriptorSets is four -- growing that count is the
    // portability risk, adding bindings to a set is not.
    if (ctx.rtEnabled) {
        VkDescriptorSetLayoutBinding bindings[2] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        // Per-instance materials, indexed by instanceCustomIndex.
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 2;
        ci.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &descSetLayout2));
    }

    // --- Pipeline layout ---
    // Raster: set0=UBO, set1=albedo, set2=normal map, set3=environment,
    //         set4=decal atlas
    // RT:     set0=UBO, set1=albedo, set2=normal map, set3=TLAS, set4=environment,
    //         set5=decal atlas
    // The environment set reuses the plain sampler layout -- it is bound to the
    // same background texture that is composited behind the scene, so metals
    // reflect the room the board is actually sitting in.
    VkDescriptorSetLayout layouts3[] = {descSetLayout0, descSetLayout1,
                                        descSetLayout1, descSetLayout1, descSetLayout1};
    // Trailing descSetLayout1 entries are the environment probe and the decal
    // atlas; both are plain sampled textures so they reuse that layout.
    VkDescriptorSetLayout layouts4[] = {descSetLayout0, descSetLayout1,
                                        descSetLayout1, descSetLayout2,
                                        descSetLayout1, descSetLayout1};

    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo plInfo = {};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    if (ctx.rtEnabled) {
        plInfo.setLayoutCount = 6;
        plInfo.pSetLayouts = layouts4;
    } else {
        plInfo.setLayoutCount = 5;
        plInfo.pSetLayouts = layouts3;
    }
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushRange;

    VK_CHECK(vkCreatePipelineLayout(ctx.device, &plInfo, nullptr, &pipelineLayout));

    // --- Descriptor pool ---
    VkDescriptorPoolSize poolSizes[4] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 128},
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 4},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
    };

    VkDescriptorPoolCreateInfo dpInfo = {};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.maxSets = 160;
    // The RT-only entries (acceleration structure, storage buffer) are last,
    // so the non-RT path counts only the first two.
    dpInfo.poolSizeCount = ctx.rtEnabled ? 4u : 2u;
    dpInfo.pPoolSizes = poolSizes;
    dpInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    VK_CHECK(vkCreateDescriptorPool(ctx.device, &dpInfo, nullptr, &descriptorPool));

    // --- Texture sampler ---
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    // samplerAnisotropy is enabled at device creation (vk_context.cpp); the
    // requested degree must not exceed the device limit.
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = std::min(16.0f, ctx.maxSamplerAnisotropy);
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;

    VK_CHECK(vkCreateSampler(ctx.device, &samplerInfo, nullptr, &textureSampler));

    // --- Graphics pipeline ---
    std::string vertPath = std::string(SHADER_DIR) + "/surface.vert.spv";
    std::string fragPath = std::string(SHADER_DIR) +
        (ctx.rtEnabled ? "/surface_rt.frag.spv" : "/surface_raster.frag.spv");

    VkShaderModule vertMod = loadShaderModule(ctx.device, vertPath);
    VkShaderModule fragMod = loadShaderModule(ctx.device, fragPath);

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    // Vertex input: pos(3) + normal(3) + uv(2) + tangent(4)
    VkVertexInputBindingDescription bindingDesc = {};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(float) * 12; // pos(3) + normal(3) + uv(2) + tangent(4)
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrDescs[4] = {};
    attrDescs[0].location = 0; attrDescs[0].format = VK_FORMAT_R32G32B32_SFLOAT;    attrDescs[0].offset = 0;
    attrDescs[1].location = 1; attrDescs[1].format = VK_FORMAT_R32G32B32_SFLOAT;    attrDescs[1].offset = 12;
    attrDescs[2].location = 2; attrDescs[2].format = VK_FORMAT_R32G32_SFLOAT;       attrDescs[2].offset = 24;
    attrDescs[3].location = 3; attrDescs[3].format = VK_FORMAT_R32G32B32A32_SFLOAT; attrDescs[3].offset = 32;

    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = 4;
    vertexInput.pVertexAttributeDescriptions = attrDescs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport = {};
    viewport.width = (float)width;
    viewport.height = (float)height;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.extent = {width, height};

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo raster = {};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample = {};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = msaaSamples;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAtt = {};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlend = {};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blendAtt;

    VkGraphicsPipelineCreateInfo pipeInfo = {};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &raster;
    pipeInfo.pMultisampleState = &multisample;
    pipeInfo.pDepthStencilState = &depthStencil;
    pipeInfo.pColorBlendState = &colorBlend;
    pipeInfo.layout = pipelineLayout;
    pipeInfo.renderPass = renderPass;
    pipeInfo.subpass = 0;

    VK_CHECK(vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline));

    // Translucent variant for dart flights. Depth writes are disabled so a
    // flight does not occlude the parts of the dart behind it, while depth
    // testing stays on so the board still hides it correctly.
    {
        VkPipelineColorBlendAttachmentState tAtt = blendAtt;
        tAtt.blendEnable = VK_TRUE;
        tAtt.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        tAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        tAtt.colorBlendOp = VK_BLEND_OP_ADD;
        tAtt.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        tAtt.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        tAtt.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo tBlend = colorBlend;
        tBlend.pAttachments = &tAtt;

        VkPipelineDepthStencilStateCreateInfo tDepth = depthStencil;
        tDepth.depthWriteEnable = VK_FALSE;

        VkGraphicsPipelineCreateInfo tInfo = pipeInfo;
        tInfo.pColorBlendState = &tBlend;
        tInfo.pDepthStencilState = &tDepth;
        VK_CHECK(vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &tInfo,
                                           nullptr, &blendPipeline));
    }

    vkDestroyShaderModule(ctx.device, vertMod, nullptr);
    vkDestroyShaderModule(ctx.device, fragMod, nullptr);

    // --- Background pipeline (fullscreen photo or Perlin noise) ---
    {
        std::string bgVertPath = std::string(SHADER_DIR) + "/background.vert.spv";
        std::string bgFragPath = std::string(SHADER_DIR) + "/background.frag.spv";

        VkShaderModule bgVertMod = loadShaderModule(ctx.device, bgVertPath);
        VkShaderModule bgFragMod = loadShaderModule(ctx.device, bgFragPath);

        VkPipelineShaderStageCreateInfo bgStages[2] = {};
        bgStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        bgStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        bgStages[0].module = bgVertMod;
        bgStages[0].pName = "main";
        bgStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        bgStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        bgStages[1].module = bgFragMod;
        bgStages[1].pName = "main";

        // Pipeline layout: set 0 = texture sampler, fragment-stage push constants
        VkPushConstantRange bgPushRange = {};
        bgPushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bgPushRange.offset = 0;
        bgPushRange.size = sizeof(BgPushConstants);

        VkPipelineLayoutCreateInfo bgPlInfo = {};
        bgPlInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        bgPlInfo.setLayoutCount = 1;
        bgPlInfo.pSetLayouts = &descSetLayout1;  // reuse texture sampler layout
        bgPlInfo.pushConstantRangeCount = 1;
        bgPlInfo.pPushConstantRanges = &bgPushRange;

        VK_CHECK(vkCreatePipelineLayout(ctx.device, &bgPlInfo, nullptr, &bgPipelineLayout));

        // No vertex input (fullscreen triangle from gl_VertexIndex)
        VkPipelineVertexInputStateCreateInfo bgVertexInput = {};
        bgVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        // No backface culling for fullscreen triangle
        VkPipelineRasterizationStateCreateInfo bgRaster = {};
        bgRaster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        bgRaster.polygonMode = VK_POLYGON_MODE_FILL;
        bgRaster.cullMode = VK_CULL_MODE_NONE;
        bgRaster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        bgRaster.lineWidth = 1.0f;

        // Depth: always pass, write enabled (so geometry can overdraw)
        VkPipelineDepthStencilStateCreateInfo bgDepth = {};
        bgDepth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        bgDepth.depthTestEnable = VK_TRUE;
        bgDepth.depthWriteEnable = VK_TRUE;
        bgDepth.depthCompareOp = VK_COMPARE_OP_ALWAYS;

        VkGraphicsPipelineCreateInfo bgPipeInfo = {};
        bgPipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        bgPipeInfo.stageCount = 2;
        bgPipeInfo.pStages = bgStages;
        bgPipeInfo.pVertexInputState = &bgVertexInput;
        bgPipeInfo.pInputAssemblyState = &inputAssembly;
        bgPipeInfo.pViewportState = &viewportState;
        bgPipeInfo.pRasterizationState = &bgRaster;
        bgPipeInfo.pMultisampleState = &multisample;
        bgPipeInfo.pDepthStencilState = &bgDepth;
        bgPipeInfo.pColorBlendState = &colorBlend;
        bgPipeInfo.layout = bgPipelineLayout;
        bgPipeInfo.renderPass = renderPass;
        bgPipeInfo.subpass = 0;

        VK_CHECK(vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &bgPipeInfo, nullptr, &bgPipeline));

        vkDestroyShaderModule(ctx.device, bgVertMod, nullptr);
        vkDestroyShaderModule(ctx.device, bgFragMod, nullptr);
    }

    // --- Segmentation pass: per-pixel class + instance ids ---
    //
    // Single sampled, unlike everything above. Ids cannot survive averaging:
    // the MSAA resolve and the kSupersample box filter both average, and the
    // mean of "dart 1" and "dart 2" is "dart 1.5" while the mean of "flight"
    // and "board" is "wire".
    {
        // Supersampled, like the colour pass. Rendering labels at output
        // resolution loses every sub-pixel structure: the spider is 0.2-0.6mm,
        // i.e. under a pixel even at 1024, so single-sample rasterisation would
        // catch it only where a triangle covers the pixel centre and leave the
        // wire class as broken speckle. The reduction back to output size
        // happens on readback by class priority, which is not averaging and so
        // keeps ids intact.
        const uint32_t segW = width;
        const uint32_t segH = height;

        VkAttachmentDescription segColorAtt = {};
        segColorAtt.format = VK_FORMAT_R8G8B8A8_UINT;
        segColorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        segColorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        segColorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        segColorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        segColorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        segColorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        segColorAtt.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        // Its own depth buffer: occlusion must match the colour image, so a
        // dart in front of the board owns those pixels rather than blending.
        VkAttachmentDescription segDepthAtt = {};
        segDepthAtt.format = VK_FORMAT_D32_SFLOAT;
        segDepthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        segDepthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        // Stored and readable: the board is planar, so depth over board pixels
        // is already implied by the homography and carries nothing new. What
        // does carry information is how far a dart stands proud of that plane,
        // which is a target no shift-invariant feature can produce.
        segDepthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        segDepthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        segDepthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        segDepthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        segDepthAtt.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        VkAttachmentReference segColorRef = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference segDepthRef = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription segSubpass = {};
        segSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        segSubpass.colorAttachmentCount = 1;
        segSubpass.pColorAttachments = &segColorRef;
        segSubpass.pDepthStencilAttachment = &segDepthRef;

        VkAttachmentDescription segAtts[] = {segColorAtt, segDepthAtt};

        VkSubpassDependency segDeps[2] = {};
        segDeps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        segDeps[0].dstSubpass = 0;
        segDeps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        segDeps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        segDeps[0].srcAccessMask = 0;
        segDeps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        // Both attachments are copied to host-visible buffers straight after
        // the pass, so the id and depth writes must reach the transfer stage.
        segDeps[1].srcSubpass = 0;
        segDeps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        segDeps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        segDeps[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
        segDeps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        segDeps[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        VkRenderPassCreateInfo segRpInfo = {};
        segRpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        segRpInfo.attachmentCount = 2;
        segRpInfo.pAttachments = segAtts;
        segRpInfo.subpassCount = 1;
        segRpInfo.pSubpasses = &segSubpass;
        segRpInfo.dependencyCount = 2;
        segRpInfo.pDependencies = segDeps;

        VK_CHECK(vkCreateRenderPass(ctx.device, &segRpInfo, nullptr, &segRenderPass));

        std::string segVertPath = std::string(SHADER_DIR) + "/segment.vert.spv";
        std::string segFragPath = std::string(SHADER_DIR) + "/segment.frag.spv";
        VkShaderModule segVertMod = loadShaderModule(ctx.device, segVertPath);
        VkShaderModule segFragMod = loadShaderModule(ctx.device, segFragPath);

        VkPipelineShaderStageCreateInfo segStages[2] = {};
        segStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        segStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        segStages[0].module = segVertMod;
        segStages[0].pName = "main";
        segStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        segStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        segStages[1].module = segFragMod;
        segStages[1].pName = "main";

        // Push constants span both stages: the vertex stage reads mvp, the
        // fragment stage reads the ids, and Vulkan requires the declared range
        // to cover every stage that references the block.
        VkPushConstantRange segPushRange = {};
        segPushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        segPushRange.offset = 0;
        segPushRange.size = sizeof(SegPushConstants);

        VkPipelineLayoutCreateInfo segPlInfo = {};
        segPlInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        segPlInfo.setLayoutCount = 0;   // no textures, no UBO: geometry and ids only
        segPlInfo.pushConstantRangeCount = 1;
        segPlInfo.pPushConstantRanges = &segPushRange;

        VK_CHECK(vkCreatePipelineLayout(ctx.device, &segPlInfo, nullptr, &segPipelineLayout));

        VkViewport segViewport = {};
        segViewport.width = (float)segW;
        segViewport.height = (float)segH;
        segViewport.maxDepth = 1.0f;

        VkRect2D segScissor = {};
        segScissor.extent = {segW, segH};

        VkPipelineViewportStateCreateInfo segViewportState = {};
        segViewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        segViewportState.viewportCount = 1;
        segViewportState.pViewports = &segViewport;
        segViewportState.scissorCount = 1;
        segViewportState.pScissors = &segScissor;

        VkPipelineMultisampleStateCreateInfo segMultisample = {};
        segMultisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        segMultisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Blending must stay off: these are integer labels, not colours.
        VkPipelineColorBlendAttachmentState segBlendAtt = {};
        segBlendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo segColorBlend = {};
        segColorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        segColorBlend.attachmentCount = 1;
        segColorBlend.pAttachments = &segBlendAtt;

        VkGraphicsPipelineCreateInfo segPipeInfo = {};
        segPipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        segPipeInfo.stageCount = 2;
        segPipeInfo.pStages = segStages;
        segPipeInfo.pVertexInputState = &vertexInput;   // same mesh layout
        segPipeInfo.pInputAssemblyState = &inputAssembly;
        segPipeInfo.pViewportState = &segViewportState;
        segPipeInfo.pRasterizationState = &raster;      // cull mode must match
        segPipeInfo.pMultisampleState = &segMultisample;
        segPipeInfo.pDepthStencilState = &depthStencil;
        segPipeInfo.pColorBlendState = &segColorBlend;
        segPipeInfo.layout = segPipelineLayout;
        segPipeInfo.renderPass = segRenderPass;
        segPipeInfo.subpass = 0;

        VK_CHECK(vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &segPipeInfo,
                                           nullptr, &segPipeline));

        vkDestroyShaderModule(ctx.device, segVertMod, nullptr);
        vkDestroyShaderModule(ctx.device, segFragMod, nullptr);
    }
}

VkDescriptorSet RenderPass::allocateUBODescriptorSet(VkContext& ctx, VkBuffer uboBuffer) {
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descSetLayout0;

    VkDescriptorSet set;
    VK_CHECK(vkAllocateDescriptorSets(ctx.device, &allocInfo, &set));

    VkDescriptorBufferInfo bufInfo = {};
    bufInfo.buffer = uboBuffer;
    bufInfo.offset = 0;
    bufInfo.range = sizeof(SceneUBO);

    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = &bufInfo;

    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    return set;
}

VkDescriptorSet RenderPass::allocateTextureDescriptorSet(VkContext& ctx, VkImageView imageView) {
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descSetLayout1;

    VkDescriptorSet set;
    VK_CHECK(vkAllocateDescriptorSets(ctx.device, &allocInfo, &set));

    VkDescriptorImageInfo imgInfo = {};
    imgInfo.sampler = textureSampler;
    imgInfo.imageView = imageView;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &imgInfo;

    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    return set;
}

void RenderPass::freeDescriptorSet(VkContext& ctx, VkDescriptorSet set) {
    vkFreeDescriptorSets(ctx.device, descriptorPool, 1, &set);
}

void RenderPass::destroy(VkContext& ctx) {
    vkDestroySampler(ctx.device, textureSampler, nullptr);
    vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    if (blendPipeline) vkDestroyPipeline(ctx.device, blendPipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);
    vkDestroyPipeline(ctx.device, bgPipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, bgPipelineLayout, nullptr);
    if (segPipeline) vkDestroyPipeline(ctx.device, segPipeline, nullptr);
    if (segPipelineLayout) vkDestroyPipelineLayout(ctx.device, segPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, descSetLayout0, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, descSetLayout1, nullptr);
    if (descSetLayout2)
        vkDestroyDescriptorSetLayout(ctx.device, descSetLayout2, nullptr);
    vkDestroyRenderPass(ctx.device, renderPass, nullptr);
    if (segRenderPass) vkDestroyRenderPass(ctx.device, segRenderPass, nullptr);
}

} // namespace dart
