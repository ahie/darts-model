#pragma once

#include <array>
#include <cmath>
#include <string>
#include <unordered_map>

namespace dart {

// ---------------------------------------------------------------------------
// Real dartboard measurements (mm) — from BDO/WDF specification
// ---------------------------------------------------------------------------

inline constexpr double INNER_BULL_R_MM   =   6.35;
inline constexpr double OUTER_BULL_R_MM   =  15.9;
// The spec gives treble and double as distances from the bull to the OUTER
// wire (107 and 170), each bed 8mm wide -- so the treble spans 99..107 and the
// double 162..170. Reading 107 as the treble's inner edge puts the whole ring
// one bed too far out.
inline constexpr double TRIPLE_INNER_R_MM =  99.0;
inline constexpr double TRIPLE_OUTER_R_MM = 107.0;
inline constexpr double TRIPLE_CENTER_R_MM= 103.0;
inline constexpr double DOUBLE_INNER_R_MM = 162.0;
inline constexpr double DOUBLE_OUTER_R_MM = 170.0;
inline constexpr double BOARD_R_MM        = 225.5;
/// Inner edge of the number ring; see RingRadii::number_ring_inner_r.
inline constexpr double NUMBER_RING_INNER_R_MM = 195.0;

// ---------------------------------------------------------------------------
// Segment layout
// ---------------------------------------------------------------------------

inline constexpr std::array<int, 20> SEGMENT_ORDER = {
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5,
};

inline constexpr double SEGMENT_ANGLE_SPAN = 2.0 * M_PI / 20.0;

// ---------------------------------------------------------------------------
// Scene geometry, in board units (BU)
// ---------------------------------------------------------------------------

/// 0.01 as the float the Blender scene stores it, widened to double.
inline constexpr double BU_PER_MM = 0.010000000507498528;

struct RingRadii {
    double inner_bull_r;
    double outer_bull_r;
    double triple_inner_r;
    double triple_outer_r;
    double triple_center_r;
    double double_inner_r;
    double double_outer_r;
    double board_r;
    /// Inner edge of the number ring.
    ///
    /// The numerals sit ON the annulus, not outside it, so the band available
    /// for printed marks is between the double and the numbers rather than the
    /// whole annulus -- which is also where the logos sit on a real board.
    ///
    /// Numeral centres are at ~2.10 BU with a half-height of ~0.072 at unit
    /// scale; NUMBER_SCALES reaches 1.75, which brings the inner edge to
    /// ~1.98. This is set just inside that so the largest numerals still clear.
    double number_ring_inner_r;
};

/// The millimetre table above, in board units. Derived rather than typed so
/// the two cannot disagree.
inline constexpr RingRadii RING_RADII_BU = {
    .inner_bull_r   = INNER_BULL_R_MM   * BU_PER_MM,
    .outer_bull_r   = OUTER_BULL_R_MM   * BU_PER_MM,
    .triple_inner_r = TRIPLE_INNER_R_MM * BU_PER_MM,
    .triple_outer_r = TRIPLE_OUTER_R_MM * BU_PER_MM,
    .triple_center_r= TRIPLE_CENTER_R_MM* BU_PER_MM,
    .double_inner_r = DOUBLE_INNER_R_MM * BU_PER_MM,
    .double_outer_r = DOUBLE_OUTER_R_MM * BU_PER_MM,
    .board_r        = BOARD_R_MM        * BU_PER_MM,
    .number_ring_inner_r = NUMBER_RING_INNER_R_MM * BU_PER_MM,
};

inline constexpr double BOARD_SURFACE_Z = 0.1899999976158142;

// ---------------------------------------------------------------------------
// Keypoint names (81 total)
// ---------------------------------------------------------------------------
//
//    0       center
//    1..20   double_<seg>         outer double ring, centred in its segment
//   21..40   triple_<seg>         inner triple ring, centred in its segment
//   41..60   double_outer_<a>_<b> outer double ring, ON the wire between a and b
//   61..80   double_inner_<a>_<b> inner double ring, ON the wire between a and b
//
// The last forty are the corners of the double beds: each of the twenty radial
// wires crosses the two circles bounding the double ring, and twenty wires
// times two radii is forty shared corners. A wire meeting a ring is a corner
// the image actually marks, unlike a point in the middle of a bed, whose
// position can only be inferred from the ring geometry around it.
//
// Every keypoint, the crossings included, is placed ON THE BOARD FACE PLANE
// (z = boardZ), not on the wire, whose centre sits 0.46-0.76mm above the face.
// That is deliberate: the app fits a planar homography to these points, and a
// label lifted off the plane would bake a view-dependent parallax into the
// fit. The cost is an offset between a crossing label and the wire's visible
// crown of up to about a pixel at the most oblique views.
//
// The crossings are appended after the first 41 keypoints, so indices 0..40
// mean the same thing in every consumer, checkpoint and annotation file.

inline constexpr int NUM_KEYPOINTS = 81;

/// Keypoint names in index order. The single source of the naming: the
/// annotator emits these, positioned per the table above.
inline std::array<std::string, NUM_KEYPOINTS> make_keypoint_names() {
    std::array<std::string, NUM_KEYPOINTS> names;
    names[0] = "center";
    for (int i = 0; i < 20; ++i) {
        names[1 + i]  = "double_" + std::to_string(SEGMENT_ORDER[i]);
        names[21 + i] = "triple_" + std::to_string(SEGMENT_ORDER[i]);

        // Wire i separates segment i from segment i+1, wrapping at the top.
        const std::string pair = std::to_string(SEGMENT_ORDER[i]) + "_" +
                                 std::to_string(SEGMENT_ORDER[(i + 1) % 20]);
        names[41 + i] = "double_outer_" + pair;
        names[61 + i] = "double_inner_" + pair;
    }
    return names;
}

// ---------------------------------------------------------------------------
// Camera defaults
// ---------------------------------------------------------------------------

/// Sensor width in mm. The focal length is quoted against this HORIZONTAL
/// aperture; see verticalFov in board_transforms.h for the vertical field.
inline constexpr float H_APERTURE_MM = 20.955f;

// ---------------------------------------------------------------------------
// Number font variants and scale options
// ---------------------------------------------------------------------------

inline constexpr int NUM_FONT_VARIANTS = 7;  // numbers_variant_0..6.glb

inline constexpr std::array<float, 6> NUMBER_SCALES = {
    1.0f, 1.1f, 1.15f, 1.25f, 1.5f, 1.75f
};

/// Spider geometry variants built at scene load.
///
/// Wire thickness and profile are baked into vertices, so they cannot be
/// randomised per frame the way a colour can. Several are built up front and
/// one is chosen per frame -- the same treatment the number fonts get.
constexpr int kSpiderVariants = 5;

/// Generated dart designs held resident. Only the selected one is ever drawn,
/// so this costs nothing per frame -- it is resident memory and one init
/// build. Measured: 64 variants are 96MB of vertex and index data and 51ms to
/// generate, plus the BLAS each needs.
///
/// The number that matters is not memory though. The renderer persists for a
/// whole run, so these are every dart the model will ever see, unlike camera,
/// lighting and placement which are continuous per frame. At 10k images an
/// epoch each design recurs ~156 times an epoch, every epoch.
constexpr int kDartVariants = 64;

/// Frames between rolling one dart design out of the pool and generating a
/// replacement. This is what makes the number of designs a run sees unbounded
/// rather than kDartVariants: without it the same 64 recur every epoch for the
/// whole run, and they are the only axis of the scene that would be discrete.
///
/// A refresh costs ~9ms, of which 6ms is the device wait and only 2.5ms is
/// the build, upload and three structure rebuilds. At 600 frames
/// that is 0.27% of throughput, and still ~17 new designs an epoch -- a few
/// thousand over a run, against a pool of 64.
///
/// The wait could be removed by deferring the frees for FRAMES_IN_FLIGHT
/// frames instead of draining, for both the mesh buffers and the retired
/// acceleration structures. Not worth the machinery at this interval.
constexpr int kDartRefreshFrames = 600;

/// Decal atlas layout: one column per character, one row per font. Must match
/// tools/generate_decal_atlas.py, which writes the same numbers to a sidecar
/// JSON beside the atlas.
constexpr int kDecalAtlasGlyphCols = 36;   // A-Z 0-9
constexpr int kDecalAtlasFontRows  = 32;

/// Longest assembled mark. Bounded by the UBO, which carries the glyph
/// sequence as two vec4s per decal.
constexpr int kDecalMaxGlyphs = 8;

/// Supersampling factor: frames are rendered at this multiple of the output
/// size and box-filtered down before readback.
///
/// 4x MSAA resolves geometry coverage but shades once per pixel, so texture
/// and shading detail still alias -- and even at a 1024 output the 0.5mm
/// spider wire is under a pixel wide. A real photograph at the same size has
/// no such aliasing, because optics, sensor and resize all filter first, so
/// the artifact is synthetic-only and sits on exactly the thin features the
/// keypoint head localises.
///
/// 2 captures nearly all of the available gain. Measured at a 512 output
/// against a 3x-supersampled reference, native rendering sits 2.4x further
/// away than 2x (mean 3.18 vs 1.34 levels, p99 41 vs 13), while 3x adds
/// almost nothing over 2x (edge energy 9.210 against 9.248). Fixed at 2: the
/// downsample is a single linear blit between SRGB images, which is an exact
/// 2x2 box average only at this factor (renderer_lib.cpp asserts it).
constexpr int kSupersample = 2;

/// Projection near/far. Named because the segmentation pass reads its depth
/// buffer back, and linearising those samples requires exactly the values the
/// projection was built with -- a copy that drifts turns depth into a plausible
/// but wrong height field, which no loss curve would reveal.
constexpr float kNearPlane = 0.1f;
constexpr float kFarPlane  = 10000.0f;

/// Per-pixel class written by the segmentation pass, for dense pretraining.
///
/// The point of the pretext task is to force spatial precision back into the
/// backbone: DINOv3's deep stages sit at 0.998 cosine under an 8px image shift,
/// so a target made of large smooth regions is solvable with exactly the blurry
/// features the task is meant to penalise. Hence the thin classes -- Wire is 0.2-0.6mm
/// and DartPoint a few pixels, and neither can be segmented without responding
/// to single-pixel translation.
///
/// Instance index rides alongside in a second channel (0 = not a dart, 1..N =
/// dart index + 1). That is what lets a per-pixel offset-to-own-tip field be
/// derived on the Python side: every dart pixel knows which tip is *its* tip,
/// which is the supervision grouped darts need to be assigned correctly and
/// which a location prior cannot supply.
///
/// Values are the contract with the Python target builder -- keep them stable.
enum class SegClass : uint8_t {
    Background = 0,
    BoardFace  = 1,
    Wire       = 2,   // spider; sub-pixel wide, deliberately included
    Numerals   = 3,
    DartPoint  = 4,   // the tip itself
    DartMetal  = 5,   // barrel and shaft
    DartFlight = 6,
    Count      = 7,
};

} // namespace dart

