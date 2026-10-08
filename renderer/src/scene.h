#pragma once

#include "vk_context.h"
#include "render_pass.h"
#include "wire_geometry.h"
#include "dart_mesh.h"

#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <unordered_map>

namespace dart {

/// Which piece of a dart a primitive belongs to.
///
/// Point is separated from Metal because a steel point is visibly darker and
/// duller than a tungsten barrel; sharing one material would make every dart
/// look machined from a single billet.
enum class DartPart { Other, Flight, Metal, Point };

struct Mesh {
    VkBuffer     vertexBuffer = VK_NULL_HANDLE;
    VmaAllocation vertexAlloc = VK_NULL_HANDLE;
    VkBuffer     indexBuffer  = VK_NULL_HANDLE;
    VmaAllocation indexAlloc  = VK_NULL_HANDLE;
    uint32_t     indexCount   = 0;
    uint32_t     vertexCount  = 0;
    glm::mat4    baseTransform = glm::mat4(1.0f);
    std::string  name;
    DartPart     dartPart = DartPart::Other;
    /// Slot in Scene::allMeshes_, which AccelStructure::blasList mirrors, so
    /// a draw can find its BLAS without searching. -1 until registered.
    int          meshIndex = -1;
};

struct Texture {
    VkImage       image     = VK_NULL_HANDLE;
    VmaAllocation alloc     = VK_NULL_HANDLE;
    VkImageView   imageView = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

/// A texture upload recorded into a caller's command buffer, and the staging
/// buffer it reads from, which must stay alive until that buffer has executed.
struct TextureUpload {
    Texture       tex;
    VkBuffer      staging      = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;
};

/// Record the upload of an 8-bit texture and the generation of its full mip
/// chain into `cmd`, leaving every level in SHADER_READ_ONLY_OPTIMAL and
/// visible to fragment shaders. `channels` of 1-3 are expanded to RGBA with
/// alpha 255. `linear` selects UNORM for data (normal maps); colour is SRGB so
/// sampling decodes it.
///
/// The full chain matters for anything read with textureLod: the environment
/// probe is sampled at LOD 5-6 for irradiance and at roughness * 6 for
/// reflections, and a texture with one level serves every one of those from
/// full-resolution texels.
TextureUpload recordTextureUpload(VkContext& ctx, RenderPass& rp, VkCommandBuffer cmd,
                                  const unsigned char* pixels, int w, int h,
                                  int channels, bool linear = false);

struct Scene {
    /// The board cylinder, generated at load rather than loaded. See
    /// board_mesh.h for why.
    Mesh boardFace;

    /// The face's roughness before the sisal layer perturbs it. Sisal is
    /// matte and wholly dielectric, so metallic is zero.
    static constexpr float kBoardFaceRoughness = 0.92f;

    /// Procedural wire frames separating the beds, one per geometry variant.
    /// Built from ringRadiiBU so the wires sit on exactly the radii the
    /// annotations and scoring use.
    ///
    /// Several are built because thickness and profile are baked into vertices
    /// and cannot vary per frame; the randomizer picks one. See kSpiderVariants.
    std::vector<Mesh> spiderVariants;

    /// The variant, clamped to what was actually built. Empty spiderVariants
    /// (no ring radii) leaves this returning nullptr so callers skip the draw.
    const Mesh* spiderFor(int variant) const {
        if (spiderVariants.empty()) return nullptr;
        if (variant < 0 || variant >= (int)spiderVariants.size()) variant = 0;
        return &spiderVariants[variant];
    }

    // Number meshes per font variant: [variant][0..19]
    std::vector<std::vector<Mesh>> numberVariants;

    /// One generated dart design: its three primitives, and the shape data the
    /// randomizer and annotator need.
    ///
    /// Only the selected variant is ever drawn, so the pool size costs
    /// resident memory and one init build and nothing per frame; see
    /// kDartVariants.
    struct DartVariant {
        std::vector<Mesh> meshes;   ///< point, barrel, tail
        DartGeometry geom;
        /// Where this variant's meshes live in allMeshes_, so a refresh can
        /// overwrite them in place and tell the accel structure which BLASes
        /// to rebuild. blasList mirrors allMeshes_ by index.
        std::vector<size_t> meshIndices;
    };
    std::vector<DartVariant> dartVariants;

    /// The variant, clamped to what was actually built.
    const DartVariant* dartFor(int variant) const {
        if (dartVariants.empty()) return nullptr;
        if (variant < 0 || variant >= (int)dartVariants.size()) variant = 0;
        return &dartVariants[variant];
    }

    /// Shape data for every variant, in variant order. The randomizer picks
    /// the variant itself -- it has to, because de-intersection depends on the
    /// shape -- so it needs the whole pool, not one entry.
    std::vector<DartGeometry> dartGeometries;

    // Textures: dummies and the decal atlas. Background photos are owned by
    // BackgroundPool (renderer_lib.h), not here.
    std::vector<Texture> textures;
    int dummyWhiteTextureIndex = -1;
    int dummyFlatNormalTextureIndex = -1;
    /// Stand-in environment probe for frames rendered without a background
    /// photo. Mid-grey rather than white: ambient is the probe's radiance
    /// times a scale, so a white probe would emit ~5x the fill a real room does
    /// and blow out every no-background frame.
    int dummyEnvTextureIndex = -1;
    /// Atlas of printed marks composited onto the board face. Required:
    /// loadAssets() aborts without it.
    int decalAtlasTextureIndex = -1;

    // BOARD_SURFACE_Z and RING_RADII_BU from constants.h, copied here by
    // loadAssets() so the geometry builders take them as plain arguments.
    double boardZ = 0.0;
    std::unordered_map<std::string, double> ringRadiiBU;
    // Dart shape data lives on DartVariant: a player throws a matched set, so
    // all three darts of a turn share one design.

    // Master list: owns all mesh GPU allocations (freed once in destroy)
    std::vector<Mesh> allMeshes_;

    void loadAssets(VkContext& ctx, RenderPass& rp, const std::string& assetDir);
    void buildDarts(VkContext& ctx, RenderPass& rp, uint32_t seed);

    /// Retire one dart design and generate a replacement in its slot.
    ///
    /// A fixed pool is every dart the model will ever see, because the
    /// renderer persists for a whole run -- unlike camera, lighting and
    /// placement, which are continuous per frame. Rolling one variant out
    /// every few hundred frames makes the effective count unbounded at small
    /// amortised cost.
    ///
    /// The caller must have waited for the device: this frees the GPU buffers
    /// the old design was drawn from. `outMeshIndices` receives the entries of
    /// allMeshes_ that changed, for AccelStructure::refreshBLAS.
    bool refreshDartVariant(VkContext& ctx, int variant, uint32_t seed,
                            std::vector<size_t>& outMeshIndices);

    void destroy(VkContext& ctx);

    /// recordTextureUpload in a one-shot command buffer, waited on.
    static Texture createTexture(VkContext& ctx, RenderPass& rp,
                                 const unsigned char* pixels, int w, int h,
                                 int channels, bool linear = false);

private:
    /// Append `mesh` to allMeshes_, recording its slot in mesh.meshIndex
    /// first so the caller's copy carries it too.
    void registerMesh(Mesh& mesh);

    /// Numeral meshes from one font variant's GLB, each placed by its node
    /// transform. Positions and normals only: numerals are drawn untextured
    /// with a flat colour and no normal map.
    void loadNumeralGLB(VkContext& ctx, const std::string& path,
                        std::vector<Mesh>& outMeshes);
    Texture createDummyWhiteTexture(VkContext& ctx, RenderPass& rp);
    void buildSpider(VkContext& ctx, RenderPass& rp);
    void buildBoard(VkContext& ctx, RenderPass& rp);
    /// A staging buffer whose copy has been recorded but not yet submitted.
    /// It has to outlive the submit, so batched uploads collect these and free
    /// them once the queue has drained.
    struct PendingStage { VkBuffer buffer; VmaAllocation alloc; };

    /// One dart design's three primitives, as Meshes. Shared by the initial
    /// pool build and by the rolling refresh, so a variant made mid-run is
    /// built exactly the way the ones made at startup were.
    void buildVariantMeshes(VkContext& ctx, VkCommandBuffer cmd,
                            const DartBuild& build, int variant,
                            std::vector<Mesh>& out,
                            std::vector<PendingStage>& pending);

    /// Record an upload into an already-open command buffer instead of
    /// submitting one of its own.
    ///
    /// uploadBuffer ends with a queue wait, so six of them -- a dart's three
    /// meshes, vertices and indices each -- would be six full round trips to
    /// the queue, which dominates the cost of a rolling refresh; the copies
    /// themselves are about 1.5MB.
    void stageBuffer(VkContext& ctx, VkCommandBuffer cmd, VkBuffer& buffer,
                     VmaAllocation& alloc, const void* data, VkDeviceSize size,
                     VkBufferUsageFlags usage,
                     std::vector<PendingStage>& pending);

    void uploadBuffer(VkContext& ctx, VkBuffer& buffer, VmaAllocation& alloc,
                      const void* data, VkDeviceSize size, VkBufferUsageFlags usage);
};

} // namespace dart
