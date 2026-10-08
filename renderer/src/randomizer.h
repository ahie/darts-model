#pragma once

#include "constants.h"

#include "dart_mesh.h"

#include <glm/glm.hpp>
#include <random>
#include <vector>

namespace dart {

/// One dart standing in the board.
///
/// x/y are the point's ENTRY coordinate on the board face, in board units --
/// the same coordinate the score zone is computed from. The pose is built to
/// put the point there; see Randomizer::randomize.
struct DartPlacement {
    float x = 0.0f;
    float y = 0.0f;

    /// Angle between the dart's axis and the board normal, radians. Sampled
    /// per dart, independent of dart index, so lean carries no information
    /// about dart count and tail offset cannot be read off the tip alone.
    float tilt = 0.0f;

    /// Direction the tail leans, radians in the board plane, 0 = +X, pi/2 = +Y.
    float azimuth = 0.0f;

    /// Rotation about the dart's own axis. Sets the flight's clock angle,
    /// independently of the lean direction.
    float roll = 0.0f;

    /// How far the point is buried, measured along the axis in board units.
    float penetration = 0.0f;
};

/// A dart's swept radius along its own axis, in world units (board units).
///
/// Built from the dart's actual vertices rather than a nominal barrel
/// diameter, because a dart is a point, a barrel, a shaft and a flight, and no
/// single radius is right for all four. A capsule of the flight's radius would
/// demand ~30mm between two darts, against a triple bed 8mm wide -- it would
/// dismantle exactly the tight groupings the skill model exists to produce.
///
/// Roll-independent by construction: the radius is the maximum over the whole
/// cross-section, so a pair that clears this test clears it at every roll. That
/// is conservative for two flights whose fins could interleave, which is the
/// right way to be wrong -- real flights in a tight group deflect each other
/// rather than passing through one another.
struct DartShape {
    static constexpr int BINS = 24;
    float axialLen = 0.0f;      ///< point tip to flight tail
    /// Max perpendicular extent of the BODY within each bin. Spans the whole
    /// dart, not just the part ahead of the flight: the shaft runs on inside
    /// the flight, and truncating the profile there would leave the rear of
    /// the shaft unchecked.
    float radius[BINS] = {};
    /// Bound on how far the flight reaches off the axis, for a cheap reject
    /// before the point test below is worth running.
    float flightReach = 0.0f;
    /// Spacing of flightPts, which bounds what the crossing test can resolve.
    float flightSampleGap = 0.0f;
    /// Axial span the flight actually occupies, for a tight broad-phase reject.
    /// Bounding it by the whole dart would give a ~108mm reject radius, which
    /// almost never fires on a grouped turn.
    float flightAxialMid = 0.0f;
    float flightAxialHalf = 0.0f;

    /// Flight vertices in the dart's own unrolled frame, decimated, scaled.
    ///
    /// The flight is the one part whose collisions depend on roll: a barrel is
    /// a solid of revolution and its swept radius is exact at every roll, while
    /// a flight is a cross of thin vanes that interleaves with another flight
    /// at a different clock angle. Kept as points rather than a radius so the
    /// test can be done at the ACTUAL roll.
    std::vector<glm::vec3> flightPts;
    /// Largest body radius anywhere on the dart, for a one-test reject.
    float maxRadius = 0.0f;
    bool valid = false;

    /// World-space start of bin *k* for a dart whose point sits at *tip* and
    /// whose axis points out of the board. Valid for k in [0, BINS], so bin k
    /// spans binStart(k)..binStart(k+1) -- a segment, not a point, which is
    /// what the overlap test needs to avoid missing contact between centres.
    glm::vec3 binStart(int k, const glm::vec3& tip,
                       const glm::vec3& axis) const {
        return tip + axis * (axialLen * (float)k / BINS);
    }
};

struct CameraState {
    glm::mat4 worldMatrix;
    glm::mat4 viewMatrix;
    /// mm, against the horizontal aperture H_APERTURE_MM.
    float focalLength;
};

struct DartMaterial {
    glm::vec3 flightColor;
    float flightRoughness;
    /// Opacity. Many flights are solid, but moulded and slim flights are often
    /// translucent, and a translucent flight tints the light passing through
    /// it -- so it casts a coloured shadow, not just a weaker one.
    float flightAlpha;

    /// Barrel. For a metal the base colour IS its F0 reflectance, so these are
    /// sampled from real finishes rather than from the RGB cube -- a uniform
    /// cube sample is mostly mid-brightness and muddy, which no metal is.
    glm::vec3 metalColor;
    float metalMetallic;
    float metalRoughness;

    /// Point, kept separate from the barrel: steel, darker and duller than
    /// tungsten, and often blackened.
    glm::vec3 pointColor;
    float pointMetallic;
    float pointRoughness;
};

struct FrameState {
    // Board
    float boardRotation = 0.0f;       // radians (numbers + face together)
    float boardFaceRotation = 0.0f;   // radians (face-only, relative to numbers)

    /// Misalignment of the number ring against the beds, radians.
    ///
    /// The ring is a separate hoop seated over the board by hand, so on a real
    /// board it never lands on an exact segment boundary -- it is a degree or
    /// two out, and stays that way until someone reseats it.
    ///
    /// This is applied to the numerals ONLY. A keypoint means "the double bed
    /// of segment 20", and that bed is fixed by the sisal and the spider; a
    /// crooked ring moves the label, not the bed. Leaving the annotation on the
    /// segments while the numerals drift is what teaches the network to locate
    /// the bed and read the numeral purely for identity, which is also what
    /// keeps scoring correct on a board whose ring is off.
    float numberRingOffset = 0.0f;

    // Camera
    CameraState camera;

    // Darts
    int numDarts = 0;
    /// How many entry points de-intersection had to move, and by how far.
    /// Reported rather than silently applied: if this fires on most frames the
    /// placement model is generating impossible turns and wants fixing at the
    /// source, not patching afterwards.
    int dartsSeparated = 0;
    float maxSeparationBU = 0.0f;
    /// Darts turned about their own axis to unpick crossed flights, and pairs
    /// no roll could separate. The second one should stay near zero; if it
    /// climbs, flights are colliding in ways rotation cannot fix and the pose
    /// distribution is worth a look.
    int dartsRerolled = 0;
    int flightsUnresolved = 0;
    std::vector<DartPlacement> darts;
    std::vector<glm::mat4> dartTransforms;  // final model matrices
    std::vector<DartMaterial> dartMaterials;

    // Light
    glm::vec3 lightPos;
    float lightIntensity;
    glm::vec3 lightColor;

    /// A second positioned fixture. Intensity 0 disables it.
    ///
    /// Positioned, not another probe entry, and that is the whole point: the
    /// environment probe is direction-only, so on a flat board every point
    /// sees the same hemisphere and the probe's contribution is spatially
    /// UNIFORM. It cannot produce a gradient across the face at any radiance.
    /// Across 49 real captures, illumination spread across same-paint beds
    /// never falls below 0.231, while one positioned light reaches as low as
    /// 0.066. A second light is what puts two overlapping gradients and two shadow
    /// directions on the surface.
    glm::vec3 light2Pos = glm::vec3(0.0f);
    float light2Intensity = 0.0f;
    glm::vec3 light2Color = glm::vec3(1.0f);
    float ambient = 0.5f;

    // Shading controls, forwarded to the fragment shader as SceneUBO.material.
    // Randomized per frame: a single fixed look would let the detector key on
    // this renderer's shading rather than on dart geometry.
    float normalStrength    = 1.0f;
    float envIntensity      = 1.0f;  // reflection probe weight

    /// The room's ceiling fixture, which the background photo cannot supply:
    /// it is 8-bit, so nothing in it exceeds 1.0, while a real fixture is one
    /// to two orders of magnitude above the wall it lights. Randomised per
    /// frame like every other lighting parameter -- venues differ in how many
    /// fittings they have and how warm they are, and a single fixed studio
    /// would be one more thing for the network to memorise.
    glm::vec3 envRoomColor = glm::vec3(1.0f);
    float envRoomRadiance = 0.0f;
    float contactAOStrength = 1.0f;  // tight AO that seats darts on the board
    float shadowStrength    = 1.0f;

    // Dartboard lighting. A single distant point light is only one of the two
    // setups that matter: most boards in use are lit by a ring fixture mounted
    // around the board itself, which illuminates head-on and largely cancels
    // its own shadows. Training on point lighting alone leaves that entire
    // regime unrepresented.
    int   lightRingMode = 0;     // 0 = distant point, 1 = ring around the board
    float lightRingRadius = 2.6f;   // BU; board radius is 2.255
    float lightRingZ = 0.9f;        // BU in front of the board face

    /// Ring emission profile: gap centre, gap half-width, asymmetry amplitude,
    /// asymmetry phase (radians). All zero is a perfect uniform ring, which no
    /// real board light is.
    glm::vec4 lightRingProfile = glm::vec4(0.0f);

    // Spider (wire frame). Boards ship with bright steel, dulled steel and
    // black-coated wire, so the colour varies rather than being fixed.
    //
    // Colour alone is not enough: a bright plated staple and a matte black
    // coating differ in how they scatter as much as in what they reflect, and
    // the coating is a dielectric over the metal rather than bare metal, so
    // metallic moves too. These three are sampled together as a finish.
    glm::vec3 wireColor = glm::vec3(0.85f, 0.84f, 0.80f);
    float wireRoughness = 0.28f;
    float wireMetallic  = 1.0f;

    /// Bed and ring colours for this frame, LINEAR, in the order the shader
    /// wants them: black bed, cream bed, ring-on-black (red), ring-on-cream
    /// (green).
    ///
    /// Randomised per frame so the palette is a property the network has to
    /// read rather than a texture it can memorise. Real boards vary a lot in
    /// TONE while never varying in hue family: the black/cream + red/green
    /// convention is what scoring depends on, so no maker departs from it,
    /// but a Target face is near-white where a yellowed club Winmau is deep
    /// cream. The analytic face in board_face.glsl makes this free.
    glm::vec3 bedBlack  = glm::vec3(0.01033f, 0.00913f, 0.00802f);
    glm::vec3 bedCream  = glm::vec3(0.80695f, 0.67244f, 0.43415f);
    glm::vec3 ringRed   = glm::vec3(0.44520f, 0.01600f, 0.01600f);
    glm::vec3 ringGreen = glm::vec3(0.01298f, 0.17465f, 0.04817f);

    /// Which spider geometry was built for this frame; see kSpiderVariants.
    int spiderVariant = 0;

    /// Which generated dart design this turn uses. One per turn, not one per
    /// dart: a player throws a matched set.
    int dartVariant = 0;

    /// Printed marks composited onto the board face, never annotated.
    ///
    /// Real boards carry brand printing and the generated face carries none; a
    /// model trained without marks reads logos as darts at full confidence.
    /// Baking fixed logos in would only teach it those logos; re-drawing a few
    /// at random placement every frame is what makes printed marks uninformative while
    /// the number ring, which is separate geometry, stays informative.
    ///
    /// Marks may land anywhere on the face, including inside the scoring area:
    /// a dart landing outside the double still has to be registered, so the
    /// lesson has to be "flat printed marks are not darts" rather than
    /// "that region is empty".
    struct Decal {
        int   fontRow = 0;            // atlas row; one font per mark
        int   glyphs[8] = {0};        // atlas columns, the assembled sequence
        int   numGlyphs = 0;          // 0 disables the slot
        float rotation = 0.0f;        // radians; local +x is the baseline
        glm::vec2 centre = glm::vec2(0.0f);  // board units
        /// Half-extents along the mark's own axes. Separate because rim
        /// printing is wide and short: the annulus is only 55.5mm across
        /// (0.555 BU), so a square mark big enough to read smears over the beds.
        float halfW = 0.0f;
        float halfH = 0.0f;
        float curvature = 0.0f;       // baseline bend, so it follows the rim
        int   style = 0;              // 0 plain, 1 outlined badge, 2 knockout
        int   shape = 0;              // 0 rounded rect, 1 ellipse
        float stroke = 0.06f;
        float opacity = 0.0f;
        glm::vec3 tint = glm::vec3(1.0f);  // linear, like every albedo
    };
    static constexpr int kMaxDecals = 5;
    Decal decals[kMaxDecals];

    /// Sisal fibre surface and wear on the board face.
    ///
    /// The analytic face is flat colour with no relief, which reads as printed
    /// card rather than as the cut ends of tightly packed sisal fibre.
    ///
    /// Generated in the shader rather than baked into a texture because wear has
    /// to differ from board to board. One baked wear pattern is a fixed image
    /// feature, and over a few hundred epochs the network learns that specific
    /// board instead of learning that boards are worn -- the same reason the
    /// printed marks are assembled per frame rather than stored.
    float fibreScale    = 28.0f;  ///< noise cells per board unit (1 BU = 100mm)
    float fibreContrast = 0.22f;  ///< how dark the gaps between fibre ends go
    float fibreRelief   = 0.35f;  ///< normal perturbation from the fibre height
    float wearAmount    = 0.0f;   ///< 0 = box-fresh board, 1 = hammered

    /// Where the board is worn. xy = centre in board units, z = falloff radius,
    /// w = weight. Placed on the same aim targets placeDarts throws at, so the
    /// wear coincides with where this renderer's darts actually land rather
    /// than sitting under a segment nothing is aimed at.
    static constexpr int kMaxWearSpots = 8;
    glm::vec4 wearSpots[kMaxWearSpots] = {};

    // Numbers
    int fontVariant = 0;
    float numberScale = 1.0f;
    bool numberMetallic = false;  // true = metallic, false = plastic

    // Background (Perlin noise)
    glm::vec3 bgColorA;
    glm::vec3 bgColorB;
    float bgNoiseScale = 4.0f;
    float bgNoiseOffsetX = 0.0f;
    float bgNoiseOffsetY = 0.0f;
    float bgIntensity = 1.0f;  // 0 = flat color, 1 = full noise contrast
};

class Randomizer {
public:
    /// `groupingProb` -- share of multi-dart turns thrown AT an existing dart
    /// rather than independently. `tightProb` -- within those, the share using
    /// the tight scatter (0.02-0.08 BU) rather than the loose one.
    ///
    /// Both default to 0.55. They are exposed because clustered darts are
    /// where the model is weakest -- tips 4-5mm out and the occasional miss --
    /// so training may want to oversample them.
    ///
    /// `panMaxDeg` / `tiltMaxDeg` -- half-widths of the uniform camera orbit.
    ///
    /// 60/48 because that reaches where the phone is. Pose is recoverable from
    /// the board keypoints alone -- the 20 double-ring points lie on one
    /// circle, so the homography is direct, and EXIF fixes the focal length
    /// exactly -- and over 49 real captures it puts the camera at a median
    /// obliquity of 42.8 degrees, reaching 67. A 45/35 orbit cannot exceed
    /// 54.5 degrees and would leave 29% of real frames unrenderable; tip error
    /// roughly doubles out there. 60/48 reproduces the measured distribution
    /// closely (p50 42.3, p90 59.2, max 70.4) without the overshoot of a wider
    /// box -- 70/55 would put the median at 48.7, training angles nobody holds.
    ///
    /// FOV needs no such widening: real frames span 15.9-17.9 degrees,
    /// entirely inside the 10-45 sampled.
    ///
    /// `dartCountWeights` -- relative weights for 0, 1, 2, 3, ... darts per
    /// frame. Empty means uniform over 0-3. Throws std::invalid_argument if
    /// any weight is negative or non-finite, or if they sum to zero.
    explicit Randomizer(uint32_t seed = 42, bool skillPlacement = true,
                        std::vector<float> dartCountWeights = {},
                        float groupingProb = 0.55f,
                        float tightProb = 0.55f,
                        float panMaxDeg = 60.0f,
                        float tiltMaxDeg = 48.0f,
                        float envRoomScale = 1.0f);

    /// `variants` is the whole pool of generated dart designs. The randomizer
    /// picks one per TURN and uses it for all three darts, because a player
    /// throws a matched set -- and it has to pick, rather than being told,
    /// because de-intersection depends on the shape and runs in here.
    ///
    /// An empty pool disables de-intersection rather than guessing a shape.
    FrameState randomize(const RingRadii& radii, double boardZ,
                         const std::vector<DartGeometry>& variants);

private:
    std::mt19937 rng_;
    bool skillPlacement_;
    float groupingProb_;
    float tightProb_;
    float panMaxDeg_;
    float tiltMaxDeg_;
    /// Multiplies the sampled ceiling-fixture radiance (env_room.glsl). 0
    /// removes the fixture and leaves the bare photo probe.
    float envRoomScale_;
    std::vector<float> dartCountCDF_;  // cumulative distribution for 0,1,2,3 darts

    /// Swept-radius profile per pool SLOT, and the DartGeometry::generation
    /// each was built from. A slot is rebuilt whenever the geometry in it
    /// carries a different generation, so a design regenerated in place can
    /// never be checked against its predecessor's shape. One entry per slot,
    /// not three: the turn's three darts share a design.
    std::vector<DartShape> dartShapes_;
    std::vector<uint64_t> dartShapeGen_;
    /// The current turn's shape, or null when there is no pool.
    const DartShape* shape_ = nullptr;

    /// Move apart any two darts whose solid geometry overlaps.
    ///
    /// Returns the largest correction applied, in board units. Entry points are
    /// the only thing adjusted: pose angles are sampled from measured
    /// distributions, and the entry coordinate is what the annotation and the
    /// score zone are both computed from, so moving it keeps every downstream
    /// label consistent by construction.
    /// inPlaneFloor caps how far a dart may be moved relative to the overlap
    /// being cleared. Only the entry point can move, so an overlap that
    /// separates along the board NORMAL has almost no in-plane component to
    /// push along and closing it that way would take a move far larger than
    /// the overlap itself. The floor refuses that; the pose pass handles those
    /// cases instead, by changing the lean, which costs no ground at all.
    float separateDarts(std::vector<DartPlacement>& darts, double boardZ,
                        float maxR, int& outMoved, float inPlaneFloor = 0.35f);

    /// Turn darts about their own axes until their flights stop crossing.
    ///
    /// Roll is the free variable here. It is already sampled uniformly and
    /// independently, so conditioning it on "the flights do not intersect"
    /// biases nothing -- it removes configurations physics never produces and
    /// leaves the rest untouched. Crucially it does NOT move the entry point,
    /// so the score zone and the grouping survive exactly, which is what
    /// separating two flights by displacement would destroy: their swept radii
    /// demand ~45mm against a triple bed 8mm wide.
    ///
    /// Returns how many darts were re-rolled. A pair that no roll can clear is
    /// left alone rather than displaced: a crossed vane is a far smaller
    /// artifact than a dart moved out of the treble it was aimed at.
    int resolveFlightRolls(std::vector<DartPlacement>& darts, double boardZ,
                           int& outUnresolved);

    float boardRotation();
    CameraState cameraOrbit(float boardRadius, float hAperture);
    std::vector<DartPlacement> placeDarts(const RingRadii& radii);
    /// Fill in tilt/azimuth/roll/penetration for one dart.
    void sampleDartPose(DartPlacement& d);
    /// Fibre parameters and the wear hotspots for one board.
    void sampleSisal(FrameState& state, const RingRadii& radii);
    /// Samples a fixture position. `panDeg`/`tiltDeg`/`dist` are reported so
    /// a second fixture can be placed RELATIVE to the first -- see the note
    /// there on why independent placement cancels its own gradient.
    /// `targetIrr` is the irradiance the fixture should DELIVER at the board;
    /// intensity is derived from it and the sampled distance. Passed in rather
    /// than drawn here so the frame's overall light level is one number, which
    /// the ambient fill is then scaled by -- see the note at its use.
    void lightPosition(float targetIrr, double boardZ, glm::vec3& pos,
                       float& intensity, float& panDeg, float& tiltDeg,
                       float& dist);
};

} // namespace dart
