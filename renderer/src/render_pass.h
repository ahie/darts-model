#pragma once

#include "vk_context.h"
#include "randomizer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace dart {

// Must match shader push_constant layout (128 bytes)
struct PushConstants {
    glm::mat4 mvp;          // 64
    glm::vec4 modelCol0;    // 16 (model X-axis, w = tx)
    glm::vec4 modelCol1;    // 16 (model Y-axis, w = ty)
    glm::vec4 modelCol2;    // 16 (model Z-axis, w = tz)
    uint32_t drawMode;      // 4  (bit31=useTexture, bit30=normalMap, bit29=decals,
                            //     bit28=translucent, bits22:15=metallic8,
                            //     bits14:7=roughness8, bits6:0=alpha7)
    float    colorR;        // 4
    float    colorG;        // 4
    float    colorB;        // 4
};
static_assert(sizeof(PushConstants) == 128);

// Pack draw mode: useTexture flag + hasNormalMap flag + metallic/roughness as 8-bit unorms
inline uint32_t packDrawMode(bool useTexture, bool hasNormalMap, float metallic, float roughness) {
    uint32_t mode = 0;
    if (useTexture) mode |= (1u << 31);
    if (hasNormalMap) mode |= (1u << 30);
    uint32_t m = static_cast<uint32_t>(std::clamp(metallic, 0.0f, 1.0f) * 255.0f + 0.5f);
    uint32_t r = static_cast<uint32_t>(std::clamp(roughness, 0.0f, 1.0f) * 255.0f + 0.5f);
    mode |= (m << 15) | (r << 7);
    return mode;
}

/// One entry per TLAS instance, indexed by instanceCustomIndex.
///
/// A ray query hands back an instance index and nothing else; this table is
/// how a reflection or shadow ray learns the colour of what it hit.
///
/// kind selects how the shader resolves the colour:
///   0  flat        -- use albedo as-is
///   1  board face  -- one instance covering twenty differently coloured beds,
///                     so albedo is meaningless and the shader evaluates
///                     boardFaceAlbedo (board_face.glsl) at the hit point
///   2  flight      -- translucent; transmit is what survives passing through
struct GpuMaterial {
    glm::vec4 albedo   = glm::vec4(0.0f);  // rgb = base colour, a = kind
    glm::vec4 transmit = glm::vec4(0.0f);  // rgb = shadow/refraction tint
};
static_assert(sizeof(GpuMaterial) == 32);

/// Format of every colour target the main pass renders into and reads back:
/// the MSAA attachment, its resolve, and the output-sized image the resolve
/// is filtered down into.
///
/// SRGB rather than UNORM so the hardware does the encoding. Shaders write
/// linear radiance and the attachment applies the exact piecewise sRGB curve
/// on store, which (a) round-trips background photos, sampled through an SRGB
/// texture, without shifting their shadows, and (b) makes blending, the MSAA
/// resolve and the supersample blit average in linear light, where averaging
/// gamma-encoded values would darken thin wires and every antialiased edge.
/// The bytes copied back to the host are still sRGB-encoded.
inline constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_SRGB;

// Must match the SceneUBO block in shaders/surface_common.glsl.
struct SceneUBO {
    // Defaults matter: a field the fill*UBO writers below forget would
    // otherwise ship uninitialised stack memory to the shader.
    glm::vec4 lightPos   = glm::vec4(0.0f);  // xyz = position, w = intensity
    glm::vec4 cameraPos  = glm::vec4(0.0f);  // xyz = position, w = ambient
    glm::vec4 lightColor = glm::vec4(1.0f);  // rgb = color, w = unused
    glm::vec4 material   = glm::vec4(1.0f, 0.8f, 1.0f, 0.95f);
                            // x = normal strength, y = env intensity,
                            // z = contact AO strength, w = shadow strength
    glm::vec4 lightRing  = glm::vec4(0.0f, 2.6f, 0.9f, 0.0f);
                            // x = mode (0 = point, 1 = ring), y = ring radius,
                            // z = ring offset along the board normal

    // Printed marks composited onto the board face. Five slots, six vec4s
    // each; the layout is documented on applyDecals in surface_common.glsl.
    glm::vec4 decal[30] = {};
    glm::vec4 decalGrid  = glm::vec4((float)kDecalAtlasGlyphCols,
                                     (float)kDecalAtlasFontRows, 0.0f, 0.0f);

    // Sisal fibre on the board face; see sisalSurface in surface_common.glsl.
    // x = noise cells per board unit, y = fibre contrast,
    // z = fibre relief, w = global wear amount.
    glm::vec4 sisal = glm::vec4(28.0f, 0.22f, 0.35f, 0.0f);
    // Wear hotspots: xy = centre in board units, z = falloff radius, w = weight.
    glm::vec4 wear[FrameState::kMaxWearSpots] = {};

    /// Inverse of the board face's model matrix. The face is one instance
    /// spanning every bed, so the shader maps a world position into the face's
    /// own frame and evaluates the bed colour from its XY.
    glm::mat4 boardFaceInvModel = glm::mat4(1.0f);

    /// The room's bright end, for reflections and ambient: rgb = the ceiling
    /// fixture's colour, w = its radiance. See shaders/env_room.glsl -- the
    /// probe is an 8-bit photograph, so nothing in it can exceed 1.0, and a
    /// real fixture sits one to two orders of magnitude above the wall it
    /// lights. That ratio is what gives metal a highlight with shape.
    ///
    /// Not the key light. That is already an analytic area source with its own
    /// GGX response; adding it here as well would double-count it.
    glm::vec4 envRoom = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);

    /// Bed and ring colours for this frame, LINEAR. Columns: black bed, cream
    /// bed, ring-on-black (red), ring-on-cream (green). Randomised per frame so
    /// training does not see one identical face; consumed by board_face.glsl.
    glm::mat4 boardPalette = glm::mat4(1.0f);

    /// Second fixture: xyz = position, w = intensity (0 disables). See
    /// FrameState::light2Pos for why this is positioned rather than another
    /// entry in the environment probe.
    glm::vec4 light2Pos = glm::vec4(0.0f);
    glm::vec4 light2Color = glm::vec4(1.0f);

    /// Ring emission profile: gap centre, gap half-width, asymmetry amplitude,
    /// asymmetry phase (radians). A uniform ring is (0, 0, 0, 0).
    glm::vec4 lightRingProfile = glm::vec4(0.0f);
};

// std140 places every member above at a 16-byte boundary with no padding,
// so the C++ size is the block size. A mismatch here means the struct and the
// GLSL block have drifted apart.
static_assert(sizeof(SceneUBO) == 912,
              "SceneUBO no longer matches surface_common.glsl's block");

// The fill*UBO writers below live next to SceneUBO deliberately, so a field
// added to the struct and its writer sit side by side. writeSceneUBO in
// renderer_lib.cpp is the one place that calls them.

/// Copy the board face's placement into the UBO.
///
/// Takes plain values rather than the Scene so that render_pass.h does not
/// have to include scene.h, which includes this header.
inline void fillBoardFaceUBO(SceneUBO& ubo, const glm::mat4& boardFaceModel) {
    ubo.boardFaceInvModel = glm::inverse(boardFaceModel);
}

/// Copy the room's bright end into the UBO.
inline void fillEnvRoomUBO(SceneUBO& ubo, const FrameState& state) {
    ubo.envRoom = glm::vec4(state.envRoomColor, state.envRoomRadiance);
}

/// Copy the second fixture and the ring emission profile into the UBO.
inline void fillLight2UBO(SceneUBO& ubo, const FrameState& state) {
    ubo.light2Pos = glm::vec4(state.light2Pos, state.light2Intensity);
    ubo.light2Color = glm::vec4(state.light2Color, 0.0f);
    ubo.lightRingProfile = state.lightRingProfile;
}

/// Copy the frame's board palette into the UBO.
inline void fillBoardPaletteUBO(SceneUBO& ubo, const FrameState& state) {
    ubo.boardPalette = glm::mat4(glm::vec4(state.bedBlack,  0.0f),
                                 glm::vec4(state.bedCream,  0.0f),
                                 glm::vec4(state.ringRed,   0.0f),
                                 glm::vec4(state.ringGreen, 0.0f));
}

/// Copy the frame's printed-mark decals into the UBO.
inline void fillDecalUBO(SceneUBO& ubo, const FrameState& state) {
    ubo.decalGrid = glm::vec4((float)kDecalAtlasGlyphCols,
                              (float)kDecalAtlasFontRows, 0.0f, 0.0f);
    for (int i = 0; i < FrameState::kMaxDecals; ++i) {
        const auto& d = state.decals[i];
        ubo.decal[i * 6 + 0] = glm::vec4((float)d.fontRow, (float)d.numGlyphs,
                                         std::cos(d.rotation), std::sin(d.rotation));
        ubo.decal[i * 6 + 1] = glm::vec4(d.centre.x, d.centre.y, d.halfW, d.halfH);
        ubo.decal[i * 6 + 2] = glm::vec4(d.tint, d.opacity);
        ubo.decal[i * 6 + 3] = glm::vec4((float)d.glyphs[0], (float)d.glyphs[1],
                                         (float)d.glyphs[2], (float)d.glyphs[3]);
        ubo.decal[i * 6 + 4] = glm::vec4((float)d.glyphs[4], (float)d.glyphs[5],
                                         (float)d.glyphs[6], (float)d.glyphs[7]);
        ubo.decal[i * 6 + 5] = glm::vec4((float)d.style, (float)d.shape,
                                         d.curvature, d.stroke);
    }
}

/// Copy the frame's sisal fibre and wear parameters into the UBO.
inline void fillSisalUBO(SceneUBO& ubo, const FrameState& state) {
    ubo.sisal = glm::vec4(state.fibreScale, state.fibreContrast,
                          state.fibreRelief, state.wearAmount);
    for (int i = 0; i < FrameState::kMaxWearSpots; ++i) {
        ubo.wear[i] = state.wearSpots[i];
    }
}

/// Must match segment.vert/frag push_constant layout (80 bytes).
///
/// Deliberately not reusing PushConstants: that carries model columns, colour
/// and a packed draw mode the segmentation pass has no use for, and at 128
/// bytes it is close enough to the 128-byte guaranteed minimum that adding ids
/// to it would risk overrunning on the low end.
struct SegPushConstants {
    glm::mat4 mvp;          // 64
    uint32_t  segClass;     // 4  dart::SegClass
    uint32_t  segInstance;  // 4  0 = not a dart, else dart index + 1
    uint32_t  pad0;         // 4
    uint32_t  pad1;         // 4
};
static_assert(sizeof(SegPushConstants) == 80);

// Must match background.frag push_constant layout (48 bytes)
struct BgPushConstants {
    glm::vec4 colorA;    // rgb = Perlin colour A (linear), w = 1 for photo mode
    glm::vec4 colorB;    // rgb = Perlin colour B (linear), w = unused
    glm::vec4 params;    // x = scale, y = offsetX, z = offsetY, w = contrast
};
static_assert(sizeof(BgPushConstants) == 48);

struct RenderPass {
    VkRenderPass     renderPass     = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline       pipeline       = VK_NULL_HANDLE;
    /// Same shaders and layout, with alpha blending and depth writes off, for
    /// translucent dart flights. A separate pipeline rather than dynamic state
    /// because blending is not dynamic in core Vulkan 1.2.
    VkPipeline       blendPipeline  = VK_NULL_HANDLE;

    VkPipelineLayout bgPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       bgPipeline       = VK_NULL_HANDLE;

    /// Segmentation pass: class + instance ids at the supersampled resolution,
    /// single sampled. Separate from the main pass because ids must not be
    /// averaged, and both the MSAA resolve and the kSupersample downsample
    /// average; the reduction to output size happens on readback by class
    /// priority.
    /// Depth-tested against its own buffer so occlusion matches the colour
    /// image -- a dart in front of the board must own those pixels.
    VkRenderPass     segRenderPass     = VK_NULL_HANDLE;
    VkPipelineLayout segPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       segPipeline       = VK_NULL_HANDLE;

    VkDescriptorSetLayout descSetLayout0 = VK_NULL_HANDLE; // UBO
    VkDescriptorSetLayout descSetLayout1 = VK_NULL_HANDLE; // Texture sampler
    VkDescriptorSetLayout descSetLayout2 = VK_NULL_HANDLE; // TLAS + materials (RT only)
    VkDescriptorPool      descriptorPool = VK_NULL_HANDLE;

    VkSampler textureSampler = VK_NULL_HANDLE;

    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_4_BIT;

    uint32_t width  = 0;
    uint32_t height = 0;

    void init(VkContext& ctx, uint32_t w, uint32_t h);
    void destroy(VkContext& ctx);

    VkDescriptorSet allocateUBODescriptorSet(VkContext& ctx, VkBuffer uboBuffer);
    VkDescriptorSet allocateTextureDescriptorSet(VkContext& ctx, VkImageView imageView);
    void freeDescriptorSet(VkContext& ctx, VkDescriptorSet set);

private:
    VkShaderModule loadShaderModule(VkDevice device, const std::string& path);
};

} // namespace dart
