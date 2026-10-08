#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <random>
#include <vector>

namespace dart {

// Darts, generated rather than loaded.
//
// Real darts vary widely in barrel, point and tail geometry, and a detector
// trained on a single dart design is noticeably weaker on sets that do not
// look like it. This generates the whole dart from measured catalogue ranges.
// The structure follows board_mesh.h and wire_geometry.h: a parameter struct,
// a sampler, and a builder that appends into interleaved vertex/index buffers.
//
// THREE PIECES, THREE MESHES. A point is dull steel, a barrel is tungsten or
// brass, and a tail is moulded plastic; they are built separately so each can
// carry its own material (see DartPart).
//
// UNITS ARE MILLIMETRES on the way in and BOARD UNITS on the way out, scaled
// by BU_PER_MM. Nothing about a dart is naturally expressed in board units,
// and every source figure behind these ranges is quoted in millimetres.
//
// AXIS: the dart runs along +Z from its point. The tip sits at the origin, so
// tipLocal is (0,0,0) and no base transform is needed.

// ---------------------------------------------------------------------------
// Barrel
// ---------------------------------------------------------------------------

/// Profile families, as (t, radius mm) control points from the point end.
///
/// Control points rather than an analytic curve because real barrels are not
/// one curve with a shape parameter: a scalloped barrel has two maxima and a
/// waist, a bomb has a bulb near the front, and no closed form covers both.
enum BarrelFamily {
    kBarrelStraight = 0,
    kBarrelPencil,
    kBarrelTorpedo,
    kBarrelBomb,
    kBarrelScallop,
    kBarrelRearHeavy,
    kBarrelTapered,
    kBarrelFamilyCount
};

/// Grip cut cross-sections. Knurl is approximated as fine rings plus a
/// roughness lift: a cross-hatch is not a solid of revolution, and at the
/// sizes this renders it reads as a finely ridged band either way.
enum GripCut {
    kCutNone = 0,
    kCutRound,     ///< rings_fine, knurl, micro
    kCutSquare,    ///< rings_bold -- a parting tool's floor with eased walls
    kCutSaw        ///< shark: steep wall on the REAR face of each crest
};

/// One gripped stretch of a barrel, in normalised barrel coordinates.
struct GripZone {
    float a = 0.0f, b = 0.0f;   ///< start/end along the barrel, 0 at the point
    float pitchMm = 0.0f;
    float depthMm = 0.0f;       ///< absolute mm: a parting tool is set to a
                                ///< depth and cuts the same on any diameter
    GripCut cut = kCutNone;
};

enum BarrelMaterial { kTungsten = 0, kBrass, kNickelSilver };

struct BarrelParams {
    BarrelFamily family = kBarrelStraight;
    float lengthMm = 50.0f;

    /// Control points after girth and jitter, (t, radius mm).
    glm::vec2 ctrl[8];
    int nCtrl = 0;

    /// Girth is SOLVED, not drawn: see sampleBarrel. Kept for reporting.
    float girth = 1.0f;

    BarrelMaterial material = kTungsten;
    float densityGcm3 = 17.0f;
    float weightG = 23.0f;
    float alloyPct = 90.0f;     ///< tungsten fraction, 0 for the alloys

    GripZone zones[8];
    int nZones = 0;
};

// ---------------------------------------------------------------------------
// Point
// ---------------------------------------------------------------------------

/// Taper shape. A point is one shank and one cone, and the whole family spread
/// is the cone's exponent, so there is no control-point list here:
///
///     r(u) = rTip + (rNeck - rTip) * (1 - u)^exponent
///
/// exponent ~1 is a straight cone and <1 a fuller convex nose. Values above 1
/// give a concave needle, which does not look like a real point, so the
/// sampler does not go there.
enum PointFinish { kSteel = 0, kBlackCoat, kGoldCoat, kSilverCoat, kGunmetal };

struct PointParams {
    float quotedMm = 35.0f;     ///< catalogue size; ~8mm of it is in the barrel
    float exposedMm = 27.0f;
    float shankMm = 12.0f;      ///< parallel section; extra length goes HERE,
                                ///< the taper stays 13-22mm whatever the size
    float taperMm = 15.0f;
    float rShank = 1.2f;
    float rNeck = 1.1f;         ///< the shank narrows slightly into the cone
    float rTip = 0.2f;
    float exponent = 1.0f;

    GripCut cut = kCutNone;     ///< shank grip: rings or a knurled band
    float gripPitchMm = 0.0f, gripDepthMm = 0.0f;
    float gripA = 0.0f, gripB = 0.0f;   ///< mm from the barrel

    PointFinish finish = kSteel;
};

// ---------------------------------------------------------------------------
// Tail: one-piece moulded shaft and flight
// ---------------------------------------------------------------------------

/// The tail is the only part with less than rotational symmetry: four vanes at
/// 90 degrees give it C4 and nothing more, so its silhouette changes with roll
/// and it cannot be built on a lathe.
///
/// It is generated as ONE closed surface, from a radius field over height and
/// angle, rather than as a stem with vanes attached:
///
///     R(z, th) = smax( core(z), vaneSection(delta, t(z), f(z), rOut(z)), k )
///
/// `vaneSection` is four plates of constant thickness meeting at 90 degrees
/// with a CONCAVE fillet in each interior corner -- both the plate and the
/// valley between vanes fall out of the cross-section, neither is a separate
/// object. Only the one-piece is generated: a two-piece shaft and flight
/// genuinely are two objects, and it is not what the harder cases look like.
struct TailParams {
    float stemMm = 27.5f;       ///< discrete sizes; ONLY this changes with
                                ///< length -- the flight is the same in every
                                ///< length of a given model
    float flightMm = 37.5f;
    float spanMm = 27.5f;       ///< tip to tip across opposite vanes
    float zFlightMm = 27.5f;    ///< where the aerofoil starts, from the barrel
    float totalMm = 65.0f;

    float r0 = 2.85f;           ///< collar radius, matched to the barrel's tail
    float r1 = 2.15f;           ///< radius where the aerofoil begins
    float spineR = 0.40f;       ///< what the round core collapses to
    float ribFrac = 0.18f;      ///< where the vanes start, as a fraction of
                                ///< the stem; ahead of it the stem is round
    float tRootMm = 1.40f;      ///< arm thickness where the ribs leave the stem
    float vaneTMm = 0.08f;      ///< the flight's own thickness
    float valleyF0 = 1.05f;     ///< concave corner fillet, shaft and tail
    float valleyF1 = 0.32f;
    float filletMm = 0.45f;     ///< soft-max blend at the vane root

    /// Kite outline control points, (u, w) with w a fraction of half-span.
    /// Four: root, two free interior, and the tip -- ONE point, on the axis,
    /// so the outline runs out to a point instead of a blunt cut.
    glm::vec2 kite[4];
    float smooth = 0.02f;       ///< corner rounding, as a blur width in u

    float rollDeg = 0.0f;       ///< clock angle of the vane cross
    glm::vec3 colorLinear = glm::vec3(0.02f);
    float flightRoughness = 0.42f;
    float flightAlpha = 1.0f;
};

// ---------------------------------------------------------------------------

struct DartParams {
    BarrelParams barrel;
    PointParams point;
    TailParams tail;
};

/// Tessellation. Defaults are sized so the finest real feature -- a 0.33mm
/// knurl pitch and a 0.08mm vane edge -- survives at the framings this
/// renders, which is 1.5-2 px/mm before the 2x supersample.
struct DartTessellation {
    int barrelSegments = 24;
    int pointSegments = 16;
    /// Axial samples per grip period. Four is enough against the supersample;
    /// more grows the barrel mesh for no visible gain.
    int gripSamplesPerPeriod = 4;
    int barrelMinRings = 96;
    int pointRings = 120;

    /// The tail's angular samples cluster toward each vane plane and again
    /// toward the 45-degree valley floor. A 0.08mm vane at 14mm radius
    /// subtends 0.33 degrees, so uniform sampling would need ~500 points
    /// around and would spend all of them on empty air.
    int tailAngularPerHalfQuadrant = 18;
    int tailAxial = 72;
};

/// One piece of a dart: interleaved position(3) normal(3) uv(2) tangent(4).
struct DartPartMesh {
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
};

/// Surface finish for the three pieces, as the draw path wants it.
///
/// It belongs to the VARIANT, not to the frame. A brass barrel is fat because
/// brass is half tungsten's density, and it is also yellow; sampling the
/// colour independently of the geometry would put a tungsten finish on a
/// brass-sized barrel and quietly undo the one axis this generator exists to
/// introduce.
struct DartFinish {
    glm::vec3 flightColor = glm::vec3(0.02f);
    float flightRoughness = 0.42f;
    /// Many flights are solid, but moulded ones are often translucent, and a
    /// translucent flight tints the light through it -- a coloured shadow,
    /// not just a weaker one.
    float flightAlpha = 1.0f;

    /// For a metal the base colour IS its F0 reflectance.
    glm::vec3 metalColor = glm::vec3(0.45f, 0.44f, 0.42f);
    float metalRoughness = 0.42f;

    glm::vec3 pointColor = glm::vec3(0.56f, 0.57f, 0.58f);
    float pointRoughness = 0.36f;
};

/// Everything about a dart's shape that is not a GPU buffer.
///
/// One per variant, not one per dart slot: a player throws a MATCHED SET, so
/// all three darts of a turn are the same design.
struct DartGeometry {
    glm::vec3 tipLocal = glm::vec3(0.0f);
    glm::vec3 tailLocal = glm::vec3(0.0f);

    /// Every vertex in the dart-local frame, for fitting an oriented box to
    /// the projected silhouette each frame.
    std::vector<glm::vec3> verts;
    /// The whole tail, stem included. Kept separate because the flight is the
    /// one part whose collisions depend on ROLL: a swept radius turns a cross
    /// of thin vanes into a solid 30mm cylinder, which would force darts ~45mm
    /// apart on two turns in three and dismantle exactly the tight groups that
    /// matter.
    std::vector<glm::vec3> flightVerts;
    /// Point and barrel.
    std::vector<glm::vec3> bodyVerts;

    DartFinish finish;

    /// Identity of this exact geometry, unique for the life of the process.
    ///
    /// Set by extractGeometry from a process-wide counter, so two extractions
    /// never share a value even when they fill the same pool slot. Anything
    /// that caches a product of the vertices (the randomizer's collision
    /// shapes) keys the cache on this, which makes a slot regenerated in place
    /// a cache miss by construction rather than by a caller remembering to
    /// invalidate. 0 means "not produced by extractGeometry" and must never be
    /// cached.
    uint64_t generation = 0;
};

/// A complete dart, in BOARD UNITS, tip at the origin, running along +Z.
struct DartBuild {
    DartPartMesh point, barrel, tail;

    glm::vec3 tipLocal = glm::vec3(0.0f);
    glm::vec3 tailLocal = glm::vec3(0.0f);

    DartFinish finish;

    /// Reporting only.
    float barrelLengthMm = 0.0f, barrelDiaMm = 0.0f, totalLengthMm = 0.0f;
};

/// Draw a complete dart. All three pieces come from one rng, in a fixed order,
/// so a seed reproduces a set exactly.
DartParams sampleDart(std::mt19937& rng);

/// Barrel mass for a profile at a given girth, grams -- the solid of
/// revolution less the two tapped holes. Exposed because the girth solve is
/// worth testing directly: it is what makes a short barrel fat and a long one
/// thin, which is the correlation the catalogue actually shows.
float barrelMassG(const glm::vec2* ctrl, int nCtrl, float lengthMm,
                  float densityGcm3, float girth);

void buildDart(const DartParams& p, const DartTessellation& tess,
               DartBuild& out);

/// Pull the Vulkan-free half of a build out, for the randomizer and the
/// annotator. Splits the vertices into body and flight along the way, and
/// stamps a fresh DartGeometry::generation.
void extractGeometry(const DartBuild& b, DartGeometry& g);

} // namespace dart
