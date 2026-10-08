#include "randomizer.h"

#include "color.h"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace dart {

namespace {

// Truncated distributions.
//
// A physical quantity sampled from a distribution and then clamped to a range
// piles every out-of-range draw onto the bound: a clamp at 2.5 sigma puts
// 0.6% of all samples at one exact value, a spike no real process produces
// and one a network can learn to key on. These resample instead, which keeps
// the in-range shape and leaves no point mass.
//
// The retry count is bounded so a range far out in the tail cannot stall a
// frame; if it is exhausted the draw falls back to uniform over the range,
// which is still inside it and still has no point mass. Every range used here
// holds most of its distribution's mass, so the fallback does not fire in
// practice.
constexpr int kTruncatedRetries = 64;

float truncatedNormal(std::mt19937& rng, float mean, float sd, float lo,
                      float hi) {
    std::normal_distribution<float> dist(mean, sd);
    for (int i = 0; i < kTruncatedRetries; ++i) {
        const float v = dist(rng);
        if (v >= lo && v <= hi) return v;
    }
    return std::uniform_real_distribution<float>(lo, hi)(rng);
}

/// Uniform over [centre - halfWidth, centre + halfWidth] intersected with
/// [lo, hi]. The intersection is sampled directly, so it needs no retries.
float truncatedUniform(std::mt19937& rng, float centre, float halfWidth,
                       float lo, float hi) {
    const float a = std::max(lo, centre - halfWidth);
    const float b = std::min(hi, centre + halfWidth);
    if (!(a < b)) return std::clamp(centre, lo, hi);
    return std::uniform_real_distribution<float>(a, b)(rng);
}

}  // namespace

Randomizer::Randomizer(uint32_t seed, bool skillPlacement,
                       std::vector<float> dartCountWeights,
                       float groupingProb, float tightProb,
                       float panMaxDeg, float tiltMaxDeg,
                       float envRoomScale)
    : rng_(seed), skillPlacement_(skillPlacement),
      groupingProb_(groupingProb), tightProb_(tightProb),
      panMaxDeg_(panMaxDeg), tiltMaxDeg_(tiltMaxDeg),
      envRoomScale_(envRoomScale) {
    // Weights for 0, 1, 2, 3, ... darts; empty is uniform over 0-3.
    if (dartCountWeights.empty()) {
        dartCountWeights = {0.25f, 0.25f, 0.25f, 0.25f};
    }
    float sum = 0.0f;
    for (size_t i = 0; i < dartCountWeights.size(); ++i) {
        const float w = dartCountWeights[i];
        if (!std::isfinite(w) || w < 0.0f) {
            throw std::invalid_argument(
                "dart_count_weights[" + std::to_string(i) + "] = " +
                std::to_string(w) +
                ": every weight must be finite and non-negative");
        }
        sum += w;
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) {
        throw std::invalid_argument(
            "dart_count_weights must have a positive, finite sum; "
            "pass an empty list for uniform over 0-3 darts");
    }
    dartCountCDF_.resize(dartCountWeights.size());
    float cumulative = 0.0f;
    for (size_t i = 0; i < dartCountWeights.size(); ++i) {
        cumulative += dartCountWeights[i] / sum;
        dartCountCDF_[i] = cumulative;
    }
    dartCountCDF_.back() = 1.0f;  // ensure no floating point gaps
}

float Randomizer::boardRotation() {
    // Boards are mounted with 20 at the top. Always -- it is the one thing
    // every venue gets right.
    //
    // In image space rotating the board is indistinguishable from rolling the
    // camera, so the in-frame angle of segment 20 is randomised by camera roll
    // (see cameraOrbit), where it physically belongs. This carries only
    // mounting error: a bracket a couple of degrees out of true, which is
    // common and visible.
    return truncatedNormal(rng_, 0.0f, glm::radians(2.5f),
                           glm::radians(-7.0f), glm::radians(7.0f));
}

CameraState Randomizer::cameraOrbit(float boardRadius, float hAperture) {
    std::uniform_real_distribution<float> panDist(-panMaxDeg_, panMaxDeg_);
    std::uniform_real_distribution<float> tiltDist(-tiltMaxDeg_, tiltMaxDeg_);

    float panDeg = panDist(rng_);
    float tiltDeg = tiltDist(rng_);
    float pan = glm::radians(panDeg);
    float tilt = glm::radians(tiltDeg);

    // Sampled as a FIELD OF VIEW, not as a focal length.
    //
    // 10-45 degrees covers the phone: its square centre crop spans about 42
    // degrees through the wide camera and 15 through the tele. (On the
    // 20.955mm aperture, 10 degrees is a ~120mm focal length.)
    //
    // Uniform in FOV rather than in focal length because FOV is the quantity
    // that matters and the two are not linearly related: uniform focal over
    // this range puts most of its mass in the narrow half.
    std::uniform_real_distribution<float> fovDist(10.0f, 45.0f);
    float fovDeg = fovDist(rng_);
    float focalLength =
        hAperture / (2.0f * std::tan(glm::radians(fovDeg) * 0.5f));

    // How much of the frame width the board spans. Can exceed 1: the board is
    // then larger than the crop and its outer edge is cut off.
    //
    // The range is measured, not reasoned: across 49 real captures -- pose
    // recovered from the board keypoints, scale from the camera distance --
    // fill runs 0.65 to 1.06 with a median of 0.80. The low end is not a cheap
    // region: tip error at fill 0.60-0.70 is about twice that at 0.90-1.10,
    // and misses roughly quadruple.
    //
    // Measure fill from the camera distance, not from the projected keypoint
    // ring. The ring foreshortens -- ~14% at 44 degrees of obliquity -- so
    // ring-derived fill moves with viewing angle and reads as a framing effect
    // that is really an angle effect.
    //
    // The upper end sets how often a dart's flight runs off the edge; over
    // this range roughly 3% of darts get a clipped flight. Only 1 of the 49
    // real frames exceeds 1.05, so that matches what the phone produces. The
    // aiming error below pushes darts toward the edges too, so the two have to
    // move together if either is revisited.
    //
    // Distance follows from the fill -- it is where the board spans `fill` of
    // the frame width -- and has to stay inside a room: 6 BU (0.6m) at the
    // near end, 55 BU (5.5m, past the oche) at the far. The formula alone is
    // happy to put a 200mm-equivalent lens 8 metres back. The fill range is
    // therefore narrowed, for this FOV, to the fills whose distance lands in
    // the room, and sampled uniformly inside that. Clamping the distance
    // instead would stack every wide-angle, high-fill draw at exactly 6 BU.
    const float fillForDist = 2.0f * boardRadius * focalLength / hAperture;
    const float fillLo = std::max(0.62f, fillForDist / 55.0f);
    const float fillHi = std::min(1.05f, fillForDist / 6.0f);
    const float fill = std::uniform_real_distribution<float>(
        fillLo, std::max(fillLo, fillHi))(rng_);
    const float dist = fillForDist / fill;

    float camX = dist * cosf(tilt) * sinf(pan);
    float camY = dist * sinf(tilt);
    float camZ = dist * cosf(tilt) * cosf(pan);

    // Where the board centre sits, as aiming error rather than as slack.
    //
    // A "keep the whole disc inside" rule cannot position it: room for the
    // centre to move is halfWidth * (1 - fill), which goes negative as soon as
    // the board overruns the crop, a common case. Modelled directly instead:
    // nobody aims a phone exactly at the bull, so the centre is displaced by a
    // fraction of the half-frame. The board disc may leave the crop as a
    // result, which is the intent.
    const float halfWidth = dist * hAperture / (2.0f * focalLength);
    // Up to 0.35 of the half-frame: median displacement 5.5% of the frame
    // with a p90 of 13%, which stops "the bull is at the centre" from being a
    // free prior the model can lean on.
    std::uniform_real_distribution<float> aimDist(0.0f, 0.35f);
    float offsetLimit = halfWidth * aimDist(rng_);

    float lx = 0.0f, ly = 0.0f;
    if (offsetLimit > 0.0f) {
        std::uniform_real_distribution<float> offDist(-offsetLimit, offsetLimit);
        lx = offDist(rng_);
        ly = offDist(rng_);
    }

    // Camera roll about the view axis.
    //
    // A phone on a tripod or in the hand sits close to upright but rarely
    // exactly there, so most of the mass is a few degrees either side. The
    // uniform tail is not decoration: with the board fixed at 20-up, a
    // roll distribution concentrated at zero would put segment 20 near the top
    // of nearly every frame, and the network could then infer keypoint
    // identity from image position instead of reading the numerals. That
    // shortcut would collapse the moment the phone was held at an angle. The
    // tail keeps the numerals load-bearing.
    float rollDeg;
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    if (u01(rng_) < 0.30f) {
        rollDeg = std::uniform_real_distribution<float>(-180.0f, 180.0f)(rng_);
    } else {
        rollDeg = truncatedNormal(rng_, 0.0f, 8.0f, -30.0f, 30.0f);
    }
    const float roll = glm::radians(rollDeg);

    glm::vec3 eye(camX, camY, camZ);
    glm::vec3 target(lx, ly, 0.0f);
    glm::vec3 forward = glm::normalize(target - eye);

    // Build rotation matrix: right, up, -forward (camera looks down -Z)
    glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::cross(forward, worldUp);
    if (glm::length(right) < 1e-6f) {
        worldUp = glm::vec3(1.0f, 0.0f, 0.0f);
        right = glm::cross(forward, worldUp);
    }
    right = glm::normalize(right);
    glm::vec3 up = glm::normalize(glm::cross(right, forward));

    // Spin the basis about the view axis. Rotating right/up rather than the
    // world-up input keeps the roll exact at every tilt, including the poles
    // where a world-up reference degenerates.
    {
        const float cr = std::cos(roll), sr = std::sin(roll);
        const glm::vec3 r0 = right, u0 = up;
        right = glm::normalize(r0 * cr + u0 * sr);
        up    = glm::normalize(u0 * cr - r0 * sr);
    }

    CameraState cam;
    cam.focalLength = focalLength;

    // World matrix. GLM columns = [right, up, -forward, eye], i.e. rows of
    // the row-vector convention.
    cam.worldMatrix = glm::mat4(
        glm::vec4(right, 0.0f),
        glm::vec4(up, 0.0f),
        glm::vec4(-forward, 0.0f),
        glm::vec4(eye, 1.0f)
    );

    // Rendering uses the standard lookAt with the rolled up vector.
    cam.viewMatrix = glm::lookAt(eye, target, up);

    return cam;
}

// ---------------------------------------------------------------------------
// Dart pose distribution
// ---------------------------------------------------------------------------
//
// A dart standing in the board has three free quantities beyond where it
// landed: how far off the board normal it leans, which way it leans, and how
// far the point is buried. All three are sampled per dart.
namespace {

// Lean off the board normal. Sisal grips the point at whatever angle the dart
// arrives at, and darts arrive on the descending part of a lob, so a dart
// perfectly square to the board is rare and a pronounced lean is common.
constexpr float kTiltMeanDeg = 18.0f;
constexpr float kTiltSdDeg   = 9.0f;
constexpr float kTiltMinDeg  =  2.0f;
constexpr float kTiltMaxDeg  = 42.0f;

// Lean direction. A dart coming down onto the board ends up tail-high, so +Y
// is the physical mode rather than an arbitrary one.
//
// This is a world-space direction, not a board-space one, which is what makes
// it meaningful: darts are placed in world coordinates and the board rotates
// underneath them, so +Y is gravity-up however the board is turned.
//
// Kept narrow: a dart leaning sideways or tail-down only happens after a
// bounce or a deflection off another dart -- most darts in a board stand
// roughly upright. A small uniform tail is kept because deflections are
// exactly the case that produces tight groups.
constexpr float kAzimuthMeanDeg = 90.0f;   // +Y, i.e. tail above the point
constexpr float kAzimuthSdDeg   = 22.0f;
constexpr float kAzimuthUniformFrac = 0.06f;

// Penetration, measured along the axis from the entry point to the tip.
// Shallow: sisal fibres close around the point and hold it with very little of
// the point buried, so a few millimetres is the normal case. The steel point is
// ~29.6mm of the model, so these depths keep the barrel clear of the face.
constexpr float kPenMeanBU = 0.070f;   //  7 mm
constexpr float kPenSdBU   = 0.030f;   //  3 mm
constexpr float kPenMinBU  = 0.025f;   //  2.5 mm — barely held
constexpr float kPenMaxBU  = 0.160f;   // 16 mm — a firm hit, still short of the barrel

}  // namespace

// Aim targets with (angle, radius) in board coordinates.
// Angle 0 = 12 o'clock (segment 20), clockwise.
struct AimTarget {
    float x, y;  // Cartesian board coords
};

static int segIndexForValue(int val) {
    for (int i = 0; i < 20; ++i) {
        if (SEGMENT_ORDER[i] == val) return i;
    }
    return 0;
}

static AimTarget segmentTarget(int segIdx, float radius) {
    // segIdx 0..19 maps to SEGMENT_ORDER, angle = segIdx * SEGMENT_ANGLE_SPAN
    float angle = static_cast<float>(segIdx * SEGMENT_ANGLE_SPAN);
    // Board convention: angle 0 = up (+Y), clockwise
    return {radius * sinf(angle), radius * cosf(angle)};
}

// --- Checkout patterns ---
// In darts you can only finish (checkout) on a double or bullseye.
// Common checkout sequences used by real players.
enum RingType { RING_TRIPLE, RING_DOUBLE, RING_SINGLE, RING_BULL };

struct CheckoutDart {
    int segVal;      // 1-20, ignored for RING_BULL
    RingType ring;
};

struct CheckoutPattern {
    int numDarts;
    CheckoutDart darts[3];
};

// Common checkout patterns (last dart is always double or bull)
static const CheckoutPattern CHECKOUT_PATTERNS[] = {
    // High checkouts (3 darts)
    {3, {{20, RING_TRIPLE}, {20, RING_TRIPLE}, {0, RING_BULL}}},     // 170
    {3, {{20, RING_TRIPLE}, {19, RING_TRIPLE}, {0, RING_BULL}}},     // 167
    {3, {{20, RING_TRIPLE}, {20, RING_TRIPLE}, {20, RING_DOUBLE}}},  // 160
    {3, {{20, RING_TRIPLE}, {19, RING_TRIPLE}, {12, RING_DOUBLE}}},  // 141
    {3, {{20, RING_TRIPLE}, {20, RING_TRIPLE}, {10, RING_DOUBLE}}},  // 140
    {3, {{19, RING_TRIPLE}, {19, RING_TRIPLE}, {12, RING_DOUBLE}}},  // 126
    {3, {{20, RING_TRIPLE}, {20, RING_SINGLE}, {20, RING_DOUBLE}}},  // 120
    {3, {{20, RING_TRIPLE}, {18, RING_TRIPLE}, {18, RING_DOUBLE}}},  // 114
    {3, {{20, RING_TRIPLE}, {14, RING_TRIPLE}, {16, RING_DOUBLE}}},  // 104
    // Medium checkouts (2 darts)
    {2, {{20, RING_TRIPLE}, {20, RING_DOUBLE}, {0, RING_BULL}}},     // 100
    {2, {{20, RING_TRIPLE}, {16, RING_DOUBLE}, {0, RING_BULL}}},     // 92
    {2, {{20, RING_TRIPLE}, {10, RING_DOUBLE}, {0, RING_BULL}}},     // 80
    {2, {{19, RING_TRIPLE}, {8, RING_DOUBLE}, {0, RING_BULL}}},      // 73
    {2, {{18, RING_TRIPLE}, {7, RING_DOUBLE}, {0, RING_BULL}}},      // 68
    {2, {{20, RING_SINGLE}, {20, RING_DOUBLE}, {0, RING_BULL}}},     // 60
    // Low checkouts (1 dart)
    {1, {{0, RING_BULL}, {0, RING_BULL}, {0, RING_BULL}}},           // 50
    {1, {{20, RING_DOUBLE}, {0, RING_BULL}, {0, RING_BULL}}},        // 40
    {1, {{18, RING_DOUBLE}, {0, RING_BULL}, {0, RING_BULL}}},        // 36
    {1, {{16, RING_DOUBLE}, {0, RING_BULL}, {0, RING_BULL}}},        // 32
    {1, {{10, RING_DOUBLE}, {0, RING_BULL}, {0, RING_BULL}}},        // 20
    {1, {{8, RING_DOUBLE}, {0, RING_BULL}, {0, RING_BULL}}},         // 16
};

static constexpr int NUM_CHECKOUT_PATTERNS =
    sizeof(CHECKOUT_PATTERNS) / sizeof(CHECKOUT_PATTERNS[0]);

static AimTarget dartToTarget(const CheckoutDart& cd, const RingRadii& radii) {
    if (cd.ring == RING_BULL) return {0.0f, 0.0f};
    float radius;
    switch (cd.ring) {
        case RING_TRIPLE: radius = static_cast<float>(radii.triple_center_r); break;
        case RING_DOUBLE: radius = static_cast<float>((radii.double_inner_r + radii.double_outer_r) * 0.5); break;
        case RING_SINGLE: radius = static_cast<float>(radii.triple_inner_r * 0.5); break;
        default: radius = 0.0f; break;
    }
    return segmentTarget(segIndexForValue(cd.segVal), radius);
}

void Randomizer::sampleDartPose(DartPlacement& d) {
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    d.tilt = truncatedNormal(rng_, glm::radians(kTiltMeanDeg),
                             glm::radians(kTiltSdDeg),
                             glm::radians(kTiltMinDeg),
                             glm::radians(kTiltMaxDeg));

    if (u01(rng_) < kAzimuthUniformFrac) {
        d.azimuth = u01(rng_) * glm::two_pi<float>();
    } else {
        std::normal_distribution<float> azDist(glm::radians(kAzimuthMeanDeg),
                                               glm::radians(kAzimuthSdDeg));
        d.azimuth = azDist(rng_);  // wrapping is irrelevant; only sin/cos is used
    }

    // Free: the flight's clock angle is independent of which way the dart
    // leans.
    d.roll = u01(rng_) * glm::two_pi<float>();

    d.penetration = truncatedNormal(rng_, kPenMeanBU, kPenSdBU, kPenMinBU,
                                    kPenMaxBU);
}

void Randomizer::sampleSisal(FrameState& state, const RingRadii& radii) {
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // Fibre coarseness. Boards differ in how tightly the bundles are packed and
    // how far back the camera is, so this varies rather than being fixed: a
    // single grain size is a signature the detector could key on. The base
    // octave lands near 3-4mm and the finest near 0.5mm, which the 2x
    // supersample resolves at ~0.55mm per shaded pixel, which is the floor on
    // how fine the grain can usefully be.
    state.fibreScale    = glm::mix(55.0f, 85.0f, u01(rng_));
    // Most of what this drives is the coarse mottle; the fine fibre octaves
    // sit at the resolution limit of the output and are faded to stop them
    // aliasing.
    state.fibreContrast = glm::mix(0.22f, 0.42f, u01(rng_));
    state.fibreRelief   = glm::mix(0.22f, 0.50f, u01(rng_));

    // Most boards in play carry visible wear; a minority are near new. Skewed
    // low so heavily-hammered boards stay the exception rather than the norm.
    float roll = u01(rng_);
    state.wearAmount = (roll < 0.18f) ? glm::mix(0.0f, 0.12f, u01(rng_))
                                      : glm::mix(0.12f, 1.0f, u01(rng_) * u01(rng_));

    // Board palette, correlated with the wear just sampled.
    //
    // A board ages as one object: the same years that hammer the treble beds
    // also yellow the cream and fade the dye out of the black, so drawing the
    // palette independently of wearAmount would produce pristine-white faces
    // riddled with dart holes. `age` therefore follows wear, loosely enough
    // that a well-kept old board and a grubby newish one both still occur.
    //
    // Authored as 8-bit sRGB, converted with srgbToLinear.
    {
        auto toLinear = [](glm::vec3 c) { return srgbToLinear(c / 255.0f); };
        // Wear contributes up to 0.7 and chance up to 0.45; the chance term's
        // range is shortened where the two together would pass 1, so a heavily
        // worn board draws its age from what is left rather than piling onto
        // exactly 1.
        const float wearAge = state.wearAmount * 0.7f;
        const float age = wearAge + u01(rng_)
                                    * std::min(0.45f, 1.0f - wearAge);
        auto jitter = [&](glm::vec3 c, float amt) {
            return c * (1.0f - amt + 2.0f * amt * u01(rng_));
        };

        // Near-white (Target, Shot) through warm cream (Winmau) to a yellowed
        // club board.
        glm::vec3 cream = glm::mix(glm::vec3(240.0f, 236.0f, 226.0f),
                                   glm::vec3(210.0f, 186.0f, 134.0f), age);
        // Charcoal through to the brown-black that dyed sisal fades toward.
        glm::vec3 black = glm::mix(glm::vec3(21.0f, 21.0f, 23.0f),
                                   glm::vec3(41.0f, 35.0f, 28.0f), age);
        // Scarlet through crimson; independent of age, it is a maker choice.
        glm::vec3 red = glm::mix(glm::vec3(198.0f, 36.0f, 34.0f),
                                 glm::vec3(148.0f, 27.0f, 31.0f), u01(rng_));
        // Grass through forest.
        glm::vec3 green = glm::mix(glm::vec3(36.0f, 134.0f, 66.0f),
                                   glm::vec3(19.0f, 90.0f, 51.0f), u01(rng_));

        state.bedCream  = toLinear(jitter(cream, 0.05f));
        state.bedBlack  = toLinear(jitter(black, 0.18f));
        state.ringRed   = toLinear(jitter(red,   0.08f));
        state.ringGreen = toLinear(jitter(green, 0.08f));
    }

    for (auto& s : state.wearSpots) s = glm::vec4(0.0f);

    const float tripleR = static_cast<float>(radii.triple_center_r);
    const float doubleR = static_cast<float>((radii.double_inner_r
                                              + radii.double_outer_r) * 0.5);

    // Slot 0 is the broad haze over the scoring area -- thousands of throws
    // that missed everything specific. Everything after it is a cluster around
    // a target players actually aim at.
    int n = 0;
    state.wearSpots[n++] = glm::vec4(0.0f, 0.0f,
                                     static_cast<float>(radii.double_outer_r) * 0.85f,
                                     glm::mix(0.10f, 0.28f, u01(rng_)));

    struct Aim { int segVal; float radius; float weight; };
    const Aim aims[] = {
        {20, tripleR, 1.00f},   // treble twenty takes the most traffic by far
        {19, tripleR, 0.55f},
        {18, tripleR, 0.35f},
        { 0, 0.0f,    0.45f},   // bull
    };
    for (const Aim& a : aims) {
        if (n >= FrameState::kMaxWearSpots) break;
        // Not every board is used the same way; skip a target sometimes.
        if (u01(rng_) < 0.15f) continue;
        AimTarget t = (a.segVal == 0)
            ? AimTarget{0.0f, 0.0f}
            : segmentTarget(segIndexForValue(a.segVal), a.radius);
        float jitter = 0.04f;
        state.wearSpots[n++] = glm::vec4(
            t.x + glm::mix(-jitter, jitter, u01(rng_)),
            t.y + glm::mix(-jitter, jitter, u01(rng_)),
            glm::mix(0.16f, 0.34f, u01(rng_)),
            a.weight * glm::mix(0.6f, 1.0f, u01(rng_)));
    }

    // A couple of doubles, from checkout attempts.
    while (n < FrameState::kMaxWearSpots && u01(rng_) < 0.55f) {
        int seg = std::uniform_int_distribution<int>(0, 19)(rng_);
        AimTarget t = segmentTarget(seg, doubleR);
        state.wearSpots[n++] = glm::vec4(t.x, t.y,
                                         glm::mix(0.12f, 0.22f, u01(rng_)),
                                         glm::mix(0.15f, 0.40f, u01(rng_)));
    }
}

namespace {

/// World-space axis of a placed dart, pointing out of the board.
glm::vec3 dartAxis(const DartPlacement& d) {
    return glm::vec3(std::sin(d.tilt) * std::cos(d.azimuth),
                     std::sin(d.tilt) * std::sin(d.azimuth),
                     std::cos(d.tilt));
}

/// World-space position of the point tip: `penetration` below the entry point,
/// measured along the axis. Mirrors the pose composition in randomize().
glm::vec3 dartTipWorld(const DartPlacement& d, double boardZ,
                       const glm::vec3& axis) {
    return glm::vec3(d.x, d.y, (float)boardZ) - axis * d.penetration;
}

/// Shortest distance between two line SEGMENTS.
///
/// Segments rather than points because a radius bin covers a slice of the dart
/// ~6mm long, and comparing bin centres as if they were spheres misses contact
/// that happens between two centres (up to ~1.8mm deep).
float segSegDistance(const glm::vec3& p0, const glm::vec3& p1,
                     const glm::vec3& q0, const glm::vec3& q1) {
    const glm::vec3 u = p1 - p0, v = q1 - q0, w = p0 - q0;
    const float a = glm::dot(u, u), b = glm::dot(u, v), c = glm::dot(v, v);
    const float d = glm::dot(u, w), e = glm::dot(v, w);
    const float denom = a * c - b * b;
    float sc, tc;
    if (denom < 1e-12f) {              // parallel: pin one end and solve
        sc = 0.0f;
        tc = (c > 1e-12f) ? e / c : 0.0f;
    } else {
        sc = (b * e - c * d) / denom;
        tc = (a * e - b * d) / denom;
    }
    sc = std::max(0.0f, std::min(1.0f, sc));
    tc = std::max(0.0f, std::min(1.0f, tc));
    // One clamp can invalidate the other, so re-solve each against the clamped
    // partner. Two passes is enough for segments.
    tc = (c > 1e-12f) ? std::max(0.0f, std::min(1.0f, (e + b * sc) / c)) : 0.0f;
    sc = (a > 1e-12f) ? std::max(0.0f, std::min(1.0f, (b * tc - d) / a)) : 0.0f;
    return glm::length(w + u * sc - v * tc);
}

/// Swept radius profile from the dart's own vertices.
///
/// The model transform is rot * diag(scale), and a rotation preserves lengths,
/// so scaling the local vertices first puts every distance below in world
/// units without needing the per-frame pose.
DartShape buildShape(const std::vector<glm::vec3>& bodyVerts,
                     const std::vector<glm::vec3>& flightVerts,
                     const glm::vec3& tipLocal, const glm::vec3& tailLocal,
                     const glm::vec3& scale) {
    DartShape sh;
    if (bodyVerts.empty()) return sh;

    const glm::vec3 tip = tipLocal * scale;
    const glm::vec3 tail = tailLocal * scale;
    const glm::vec3 along = tail - tip;
    const float len = glm::length(along);
    if (!(len > 1e-6f)) return sh;
    const glm::vec3 axis = along / len;

    sh.axialLen = len;

    // Decimated to bound the pairwise cost: the test is O(n^2) in these and
    // runs per pair per candidate roll. A vane is a flat sheet, so a coarse
    // sampling of it still detects two of them crossing.
    constexpr size_t kMaxFlightPts = 56;
    const size_t stride =
        flightVerts.empty() ? 1
                            : std::max<size_t>(1, flightVerts.size() / kMaxFlightPts);
    for (size_t k = 0; k < flightVerts.size(); k += stride) {
        const glm::vec3 rel = flightVerts[k] * scale - tip;
        sh.flightPts.push_back(rel);
        const float t = glm::dot(rel, axis);
        sh.flightReach = std::max(sh.flightReach, glm::length(rel - axis * t));
    }
    // How far apart the samples are, which is what bounds the test's
    // resolution: two vanes that cross have samples within roughly this of
    // each other, and no closer. Comparing against the vane's MATERIAL
    // thickness instead would miss most crossings. Deliberately generous: the remedy is a
    // rotation that moves nothing, so over-detecting costs nothing at all,
    // while under-detecting leaves vanes visibly passing through each other.
    if (sh.flightPts.size() > 1) {
        const float area = 3.14159f * sh.flightReach * sh.flightReach;
        sh.flightSampleGap =
            1.5f * std::sqrt(area / (float)sh.flightPts.size());
        float lo = 1e9f, hi = -1e9f;
        for (const glm::vec3& q : sh.flightPts) {
            const float t = glm::dot(q, axis);
            lo = std::min(lo, t);
            hi = std::max(hi, t);
        }
        sh.flightAxialMid = 0.5f * (lo + hi);
        sh.flightAxialHalf = 0.5f * (hi - lo);
    }

    for (const glm::vec3& v : bodyVerts) {
        const glm::vec3 rel = v * scale - tip;
        const float t = glm::dot(rel, axis);
        const float r = glm::length(rel - axis * t);
        // Clamped rather than dropped: a vertex fractionally beyond either end
        // still belongs to the end cap, and discarding it would understate the
        // radius exactly at the point and the flight tail.
        int bin = (int)((t / len) * DartShape::BINS);
        bin = std::max(0, std::min(DartShape::BINS - 1, bin));
        sh.radius[bin] = std::max(sh.radius[bin], r);
    }
    // A bin with no vertices would read as zero radius and let another dart
    // pass straight through it, so carry the neighbouring radius across.
    for (int k = 1; k < DartShape::BINS; ++k)
        if (sh.radius[k] <= 0.0f) sh.radius[k] = sh.radius[k - 1];
    for (int k = DartShape::BINS - 2; k >= 0; --k)
        if (sh.radius[k] <= 0.0f) sh.radius[k] = sh.radius[k + 1];

    for (int k = 0; k < DartShape::BINS; ++k)
        sh.maxRadius = std::max(sh.maxRadius, sh.radius[k]);

    sh.valid = true;
    return sh;
}

}  // namespace

namespace {

/// Deepest overlap between two dart BODIES, or 0 if they are clear.
///
/// One shape for both: the darts of a turn are a matched set.
/// `outDir` receives the direction to push *a* away from *b*.
float bodyPenetration(const DartShape& sh,
                      const DartPlacement& a, const DartPlacement& b,
                      double boardZ, glm::vec3* outDir) {
    const glm::vec3 ai = dartAxis(a), aj = dartAxis(b);
    const glm::vec3 ti = dartTipWorld(a, boardZ, ai);
    const glm::vec3 tj = dartTipWorld(b, boardZ, aj);

    // One segment test against the whole dart before the 576 that follow. Two
    // darts anywhere but on top of each other are rejected here, which is the
    // overwhelming majority of the calls: this runs per pair per pass of the
    // separation loop, and again for every pose candidate being vetted.
    if (segSegDistance(ti, ti + ai * sh.axialLen, tj, tj + aj * sh.axialLen)
        > 2.0f * sh.maxRadius) {
        return 0.0f;
    }

    // The radius profile covers the BODY only -- point, barrel and shaft, all
    // solids of revolution a swept radius describes exactly. It spans the
    // dart's whole length even so, because the shaft runs on inside the
    // flight. A flight's swept radius would make it a solid cylinder 32mm
    // across and demand ~45mm of displacement on two turns in three; crossed
    // flights are unpicked by turning the dart instead, which costs no ground.
    //
    // Closest approach of the two AXES is not enough on its own: with a radius
    // that varies along the dart, the worst overlap can sit well away from
    // where the axes are nearest.
    float bestPen = 0.0f;
    for (int k = 0; k < DartShape::BINS; ++k) {
        const glm::vec3 k0 = sh.binStart(k, ti, ai);
        const glm::vec3 k1 = sh.binStart(k + 1, ti, ai);
        for (int l = 0; l < DartShape::BINS; ++l) {
            const glm::vec3 l0 = sh.binStart(l, tj, aj);
            const glm::vec3 l1 = sh.binStart(l + 1, tj, aj);
            const float d = segSegDistance(k0, k1, l0, l1);
            const float pen = sh.radius[k] + sh.radius[l] - d;
            if (pen > bestPen) {
                bestPen = pen;
                if (outDir) {
                    // Between the bin midpoints: it only has to say which way
                    // to push, and the exact closest points would give a
                    // near-identical bearing.
                    const glm::vec3 delta = (k0 + k1) * 0.5f - (l0 + l1) * 0.5f;
                    const float dl = glm::length(delta);
                    *outDir = (dl > 1e-6f) ? delta / dl
                                           : glm::vec3(1.0f, 0.0f, 0.0f);
                }
            }
        }
    }
    return bestPen;
}

}  // namespace

float Randomizer::separateDarts(std::vector<DartPlacement>& darts,
                                double boardZ, float maxR, int& outMoved,
                                float inPlaneFloor) {
    outMoved = 0;
    const int n = (int)darts.size();
    if (n < 2 || shape_ == nullptr || !shape_->valid) return 0.0f;
    const DartShape& sh = *shape_;

    // Iterated because resolving one pair can push a dart into a third. The
    // pass cap is far more than the geometry needs -- three darts have three
    // pairs -- and the loop exits as soon as a pass finds nothing.
    constexpr int MAX_PASSES = 64;
    // Leaves the surfaces just clear of touching rather than exactly tangent.
    // 0.0002 BU is 0.02mm, invisible at any resolution this renders at.
    constexpr float kEps = 0.0002f;

    float worst = 0.0f;
    std::vector<bool> moved(n, false);

    for (int pass = 0; pass < MAX_PASSES; ++pass) {
        bool any = false;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                glm::vec3 bestDir(0.0f);
                const float bestPen = bodyPenetration(sh, darts[i], darts[j],
                                                      boardZ, &bestDir);
                if (bestPen <= 0.0f) continue;

                // Only the entry point can move, so the correction is applied
                // in the board plane. A near-vertical separation direction has
                // little to push along, hence the fallback.
                glm::vec2 dir(bestDir.x, bestDir.y);
                float dlen = glm::length(dir);
                if (dlen < 1e-4f) {
                    const glm::vec2 gap(darts[i].x - darts[j].x,
                                        darts[i].y - darts[j].y);
                    dlen = glm::length(gap);
                    dir = (dlen > 1e-6f) ? gap / dlen : glm::vec2(1.0f, 0.0f);
                } else {
                    dir /= dlen;
                }

                // Split the correction between the pair so neither is singled
                // out, and divide by the in-plane component: pushing along the
                // board plane only closes the part of the gap that lies in it.
                const float inPlane = std::max(dlen, inPlaneFloor);
                const float step = (bestPen * 0.5f + kEps) / inPlane;

                darts[i].x += dir.x * step;
                darts[i].y += dir.y * step;
                darts[j].x -= dir.x * step;
                darts[j].y -= dir.y * step;

                for (int idx : {i, j}) {
                    const float r = std::sqrt(darts[idx].x * darts[idx].x +
                                              darts[idx].y * darts[idx].y);
                    if (r > maxR && r > 1e-6f) {
                        darts[idx].x *= maxR / r;
                        darts[idx].y *= maxR / r;
                    }
                    moved[idx] = true;
                }
                worst = std::max(worst, step);
                any = true;
            }
        }
        if (!any) break;
    }

    for (int i = 0; i < n; ++i) if (moved[i]) ++outMoved;
    return worst;
}

int Randomizer::resolveFlightRolls(std::vector<DartPlacement>& darts,
                                   double boardZ, int& outUnresolved) {
    outUnresolved = 0;
    const int n = (int)darts.size();
    if (n < 2 || shape_ == nullptr || !shape_->valid ||
        shape_->flightPts.empty()) {
        return 0;
    }
    const DartShape& sh = *shape_;

    constexpr int kCandidates = 8;          // roll offsets tried per dart

    // Every candidate below perturbs roll, azimuth or tilt -- and NOT x or y.
    // That is the whole point: the entry coordinate is what the score zone and
    // every annotation are computed from, so leaving it alone keeps the labels
    // and the grouping exactly as sampled. Roll turns the vanes about the
    // dart's own axis; azimuth and tilt swing the TAIL, with the dart's full
    // length as the lever arm, so a few degrees moves a flight much further
    // than any rotation of the vanes can. It is also what really happens: two
    // darts arriving at the same spot deflect each other's lean, they do not
    // land on top of one another.
    auto flightWorld = [&](const DartPlacement& d,
                           std::vector<glm::vec3>& out) {
        const glm::vec3 axis = dartAxis(d);
        const glm::vec3 tip = dartTipWorld(d, boardZ, axis);
        // Same frame the pose composition builds, so a point placed here lands
        // exactly where the rendered vertex will.
        const glm::vec3 ref = (std::abs(axis.z) < 0.999f)
                                  ? glm::vec3(0.0f, 0.0f, 1.0f)
                                  : glm::vec3(1.0f, 0.0f, 0.0f);
        const glm::vec3 tangent = glm::normalize(glm::cross(ref, axis));
        const glm::mat3 frame(tangent, glm::cross(axis, tangent), axis);
        const float cr = std::cos(d.roll), sr = std::sin(d.roll);
        const glm::mat3 rollZ(glm::vec3(cr, sr, 0.0f),
                              glm::vec3(-sr, cr, 0.0f),
                              glm::vec3(0.0f, 0.0f, 1.0f));
        const glm::mat3 rot = frame * rollZ;
        out.clear();
        out.reserve(sh.flightPts.size());
        for (const glm::vec3& q : sh.flightPts) out.push_back(tip + rot * q);
    };

    // Squared throughout: the only comparisons are against each other and
    // against a threshold, and a square root per pair is most of the work.
    // `floor` lets a caller that only needs "is this below X" stop early --
    // which is the common case, since a candidate that is still colliding
    // usually proves it within the first few points.
    auto minDist2 = [](const std::vector<glm::vec3>& a,
                       const std::vector<glm::vec3>& b, float floor2) {
        float best = 1e9f;
        for (const glm::vec3& p : a) {
            for (const glm::vec3& q : b) {
                const glm::vec3 d = p - q;
                const float d2 = d.x * d.x + d.y * d.y + d.z * d.z;
                if (d2 < best) {
                    best = d2;
                    if (best <= floor2) return best;
                }
            }
        }
        return best;
    };

    std::vector<glm::vec3> wa, wb;
    int rerolled = 0;

    // Passes, because moving one dart's tail out of another's flight can put it
    // into a third's.
    for (int pass = 0; pass < 2; ++pass) {
    outUnresolved = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // Broad phase on the flight centres, so the O(n^2) point test only
            // runs for pairs that could plausibly touch.
            const glm::vec3 ai = dartAxis(darts[i]), aj = dartAxis(darts[j]);
            const glm::vec3 ci = dartTipWorld(darts[i], boardZ, ai)
                               + ai * sh.flightAxialMid;
            const glm::vec3 cj = dartTipWorld(darts[j], boardZ, aj)
                               + aj * sh.flightAxialMid;
            const float reach = 2.0f * (sh.flightReach + sh.flightAxialHalf
                                        + sh.flightSampleGap);
            if (glm::length(ci - cj) > reach) continue;

            // A vane is a thin sheet sampled at a finite spacing, so the
            // threshold is what the sampling can resolve, not the polyester's
            // 0.3mm. See DartShape::flightSampleGap.
            const float touch = sh.flightSampleGap;
            const float touch2 = touch * touch;

            // Body overlaps get here too, not only crossed flights.
            // Displacement alone cannot clear every body: when two darts
            // overlap along the board NORMAL, the in-plane component of the
            // separation is tiny and closing the gap that way would need a
            // move far larger than the overlap. Changing the lean fixes those
            // for nothing, since it does not move the entry point.
            const float startBodyPen =
                bodyPenetration(sh, darts[i], darts[j], boardZ, nullptr);

            flightWorld(darts[i], wa);
            flightWorld(darts[j], wb);
            const bool flightsClose = minDist2(wa, wb, touch2) <= touch2;
            if (!flightsClose && startBodyPen <= 0.0f) continue;

            // Try j first and only then i, so a frame never has both darts
            // disturbed when moving one was enough.
            bool cleared = false;
            for (int who : {j, i}) {
                const int other = (who == j) ? i : j;
                flightWorld(darts[other], (who == j) ? wa : wb);
                std::vector<glm::vec3>& mine = (who == j) ? wb : wa;
                const DartPlacement original = darts[who];
                DartPlacement best = original;
                float bestGap = -1.0f;

                // Azimuth and tilt swing the whole dart, not just its flight,
                // so a pose that parts two flights can push two BODIES into
                // each other that the body pass had already cleared, by
                // several millimetres. Candidates are
                // therefore vetted against every other dart's body, and the
                // starting pose is always body-clear because the body pass ran
                // first, so rejecting all of them is safe.
                auto worstBodyPen = [&](const DartPlacement& cand) {
                    float worst = 0.0f;
                    for (int o = 0; o < n; ++o) {
                        if (o == who) continue;
                        worst = std::max(
                            worst,
                            bodyPenetration(sh, cand, darts[o], boardZ,
                                            nullptr));
                    }
                    return worst;
                };
                // Solid bodies first, always: a crossed vane is a small
                // artifact, a barrel through a barrel is an impossible object.
                float bestPen = worstBodyPen(original);
                auto better = [&](float pen, float gap) {
                    if (pen < bestPen - 1e-7f) return true;   // less overlap
                    if (pen > bestPen + 1e-7f) return false;  // more overlap
                    return gap > bestGap;                     // same, wider gap
                };

                // Roll alone first: it changes only which way the vanes point,
                // leaving the dart's silhouette and lean untouched. Azimuth and
                // tilt come after, because they do alter the pose -- by a few
                // degrees, within the spread the pose sampler already draws
                // from -- and are only worth spending when rolling cannot cope.
                for (int c = 0; c < kCandidates && !cleared; ++c) {
                    DartPlacement cand = original;
                    cand.roll = original.roll
                              + 6.2831853f * (c + 1) / (kCandidates + 1);
                    // Roll cannot move the body, so its penetration is
                    // whatever the starting pose already had.
                    darts[who] = cand;
                    flightWorld(cand, mine);
                    const float gap = minDist2(wa, wb, bestGap);
                    if (better(bestPen, gap)) { bestGap = gap; best = cand; }
                    if (gap > touch2 && bestPen <= 0.0f) cleared = true;
                }
                if (!cleared) {
                    constexpr float kAz[] = {0.07f, -0.07f, 0.15f, -0.15f,
                                             0.26f, -0.26f};
                    constexpr float kTilt[] = {0.0f, 0.05f, -0.05f};
                    for (float dt : kTilt) {
                        for (float da : kAz) {
                            DartPlacement cand = best;
                            cand.azimuth = original.azimuth + da;
                            // Kept off vertical and off the board face; the
                            // sampler's own range is well inside this.
                            cand.tilt = std::max(0.02f,
                                                 std::min(1.20f,
                                                          original.tilt + dt));
                            darts[who] = cand;
                            flightWorld(cand, mine);
                            // Floor only while optimising the gap; with
                            // an overlap outstanding the true gap is
                            // needed for the tie-break.
                            const float gap = minDist2(
                                wa, wb, bestPen <= 0.0f ? bestGap : 0.0f);
                            // The body test is the expensive one, so it is
                            // skipped for a candidate that cannot win on the
                            // flight gap and has no overlap to improve on.
                            if (bestPen <= 0.0f && gap <= bestGap) continue;
                            const float pen = worstBodyPen(cand);
                            if (!better(pen, gap)) continue;
                            bestPen = pen;
                            bestGap = gap;
                            best = cand;
                            if (gap > touch2 && pen <= 0.0f) {
                                cleared = true;
                                break;
                            }
                        }
                        if (cleared) break;
                    }
                }
                darts[who] = best;
                flightWorld(best, mine);
                if (bestGap > 0.0f) ++rerolled;
                if (cleared) break;
            }
            if (!cleared) ++outUnresolved;
        }
    }
    if (outUnresolved == 0) break;
    }
    return rerolled;
}

std::vector<DartPlacement> Randomizer::placeDarts(const RingRadii& radii) {
    // Sample dart count from CDF [0, 1, 2, 3]
    float countRoll = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_);
    int numDarts = 0;
    for (size_t i = 0; i < dartCountCDF_.size(); ++i) {
        if (countRoll < dartCountCDF_[i]) { numDarts = static_cast<int>(i); break; }
    }
    if (numDarts == 0) return {};

    float maxR = static_cast<float>(radii.board_r) * 0.95f;
    std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);

    // --- Simple random placement (no skill model) ---
    if (!skillPlacement_) {
        float sigma = static_cast<float>(radii.double_outer_r) * 0.4f;
        std::normal_distribution<float> gaussDist(0.0f, sigma);
        std::vector<DartPlacement> result;
        result.reserve(numDarts);
        for (int i = 0; i < numDarts; ++i) {
            DartPlacement d;
            while (true) {
                d.x = gaussDist(rng_);
                d.y = gaussDist(rng_);
                if (sqrtf(d.x * d.x + d.y * d.y) < maxR) break;
            }
            sampleDartPose(d);
            result.push_back(d);
        }
        return result;
    }

    // --- Sample skill level ---
    // 0=pro, 1=good, 2=average, 3=beginner
    // Weights: 10% pro, 25% good, 40% average, 25% beginner
    float skillRoll = unitDist(rng_);
    int skill;
    float sigma;
    if (skillRoll < 0.10f)      { skill = 0; sigma = 0.08f; }   // pro
    else if (skillRoll < 0.35f) { skill = 1; sigma = 0.18f; }   // good
    else if (skillRoll < 0.75f) { skill = 2; sigma = 0.40f; }   // average
    else                        { skill = 3; sigma = 0.75f; }   // beginner

    // --- Sample a random per-player bias (drift) ---
    // Small systematic offset simulating a player's consistent miss direction
    std::normal_distribution<float> biasDist(0.0f, sigma * 0.3f);
    float biasX = biasDist(rng_);
    float biasY = biasDist(rng_);

    // --- Check if this is a checkout turn ---
    // Checkout = finishing a leg, last dart must be double or bull.
    // Pro/good attempt checkouts more often than average/beginner.
    //
    // Grouping is decided BEFORE the checkout roll. A checkout aims the three
    // darts at three different targets, so any turn that becomes a checkout
    // can never be a group; rolling checkout first would take 30% of exactly
    // the turns most likely to group -- the pro and good players -- and make
    // three-dart treble groups very rare.
    const bool groupingTurn = (numDarts >= 2) && (unitDist(rng_) < groupingProb_);

    float checkoutChance = (skill <= 1) ? 0.30f : (skill == 2) ? 0.15f : 0.05f;
    const bool isCheckout = !groupingTurn && (unitDist(rng_) < checkoutChance);

    if (isCheckout) {
        // Only patterns thrown with exactly the drawn number of darts, so the
        // configured dart-count weights hold for checkout turns too. A count
        // no pattern covers (more than three darts) is thrown as an ordinary
        // scoring turn below.
        std::vector<int> eligible;
        for (int p = 0; p < NUM_CHECKOUT_PATTERNS; ++p) {
            if (CHECKOUT_PATTERNS[p].numDarts == numDarts) {
                eligible.push_back(p);
            }
        }
        if (!eligible.empty()) {
            int pick = std::uniform_int_distribution<int>(0, (int)eligible.size() - 1)(rng_);
            const auto& pattern = CHECKOUT_PATTERNS[eligible[pick]];

            std::vector<DartPlacement> result;
            result.reserve(numDarts);
            std::normal_distribution<float> noiseDist(0.0f, sigma);

            for (int i = 0; i < numDarts; ++i) {
                AimTarget target = dartToTarget(pattern.darts[i], radii);
                DartPlacement d;
                while (true) {
                    d.x = target.x + noiseDist(rng_) + biasX;
                    d.y = target.y + noiseDist(rng_) + biasY;
                    if (sqrtf(d.x * d.x + d.y * d.y) < maxR) break;
                }
                sampleDartPose(d);
                result.push_back(d);
            }
            return result;
        }
    }

    // --- Scoring turn (not a checkout) ---
    float tripleR = static_cast<float>(radii.triple_center_r);
    float doubleR = static_cast<float>((radii.double_inner_r + radii.double_outer_r) * 0.5);
    float singleR = static_cast<float>(radii.triple_inner_r * 0.5);

    int seg20 = segIndexForValue(20);
    int seg19 = segIndexForValue(19);
    int seg18 = segIndexForValue(18);

    auto pickScoringTarget = [&]() -> AimTarget {
        float roll = unitDist(rng_);
        if (skill <= 1) {
            // Pro/good: mostly T20, T19, T18, bull
            if (roll < 0.40f) return segmentTarget(seg20, tripleR);
            if (roll < 0.65f) return segmentTarget(seg19, tripleR);
            if (roll < 0.80f) return segmentTarget(seg18, tripleR);
            if (roll < 0.90f) return {0.0f, 0.0f};  // bullseye
            // Random double (checkout practice)
            int seg = std::uniform_int_distribution<int>(0, 19)(rng_);
            return segmentTarget(seg, doubleR);
        } else if (skill == 2) {
            // Average: mix of T20, T19, random singles
            if (roll < 0.30f) return segmentTarget(seg20, tripleR);
            if (roll < 0.45f) return segmentTarget(seg19, tripleR);
            if (roll < 0.55f) return {0.0f, 0.0f};  // bullseye
            // Random segment, single area
            int seg = std::uniform_int_distribution<int>(0, 19)(rng_);
            return segmentTarget(seg, singleR);
        } else {
            // Beginner: aim roughly at center or random
            if (roll < 0.40f) return {0.0f, 0.0f};  // trying for bull
            int seg = std::uniform_int_distribution<int>(0, 19)(rng_);
            float r = unitDist(rng_) * static_cast<float>(radii.double_outer_r);
            return segmentTarget(seg, r);
        }
    };

    // All darts in a scoring turn usually aim at the same target
    AimTarget mainTarget = pickScoringTarget();

    // Grouping turn: every dart at one target, tightly, with no switching.
    //
    // A tight group is the case the detector most needs and most often gets
    // wrong, yet the skill model alone rarely produces one (about 1% of
    // frames): the checkout chance sends the three darts at three different
    // targets, and the 20% per-dart mid-turn switch scatters the rest.
    //
    // Rather than widen the skill sigmas, which would also blur the sparse
    // case, this turn type behaves like a good player repeating one target.
    // The scatter is drawn small directly instead of taken from the skill
    // tier, since the point is to populate the tight end of the distribution
    // rather than to model an average player.
    float turnSigma = sigma;
    float turnBiasX = biasX, turnBiasY = biasY;
    if (groupingTurn) {
        // What matters is three darts CLOSE TO a target, not three darts
        // scoring it. All three inside the treble bed is a 180 -- rarer than
        // the case the detector actually struggles with, which is a tight
        // group sitting on T20 and spilling into S20 or the neighbouring
        // beds. Chasing the strict version would push the scatter down to a
        // few millimetres and make the data less representative, not more.
        //
        // So: a tight mode that lands the group on the target, and a looser
        // one that spreads it across the beds around it. Both are groups; both
        // are hard; only the second is common in real play.
        turnSigma = (unitDist(rng_) < tightProb_)
                      ? glm::mix(0.02f, 0.08f, unitDist(rng_))
                      : glm::mix(0.08f, 0.18f, unitDist(rng_));

        // Bias has to come from the turn's own scatter, not the skill tier.
        // At 0.3x the skill sigma a beginner grouping turn would get a 2mm
        // group displaced by a 22mm bias -- tight, but nowhere near what it
        // was aimed at, which defeats the point of aiming at T20.
        std::normal_distribution<float> turnBias(0.0f, turnSigma * 0.5f);
        turnBiasX = turnBias(rng_);
        turnBiasY = turnBias(rng_);

        // Players group where they aim, and they aim at T20 far more than
        // anywhere else. The generic scoring picker spreads across T20/T19/T18/
        // bull by skill tier, which is right for an ordinary turn but wastes
        // most grouping turns on targets the app will rarely be pointed at.
        float t = unitDist(rng_);
        if      (t < 0.62f) mainTarget = segmentTarget(seg20, tripleR);
        else if (t < 0.78f) mainTarget = segmentTarget(seg19, tripleR);
        else if (t < 0.86f) mainTarget = segmentTarget(seg18, tripleR);
        else if (t < 0.93f) mainTarget = AimTarget{0.0f, 0.0f};   // bull
        // the remainder keeps whatever the generic picker chose
    }

    std::vector<DartPlacement> result;
    result.reserve(numDarts);

    std::normal_distribution<float> noiseDist(0.0f, turnSigma);

    for (int i = 0; i < numDarts; ++i) {
        // 20% chance of switching target mid-turn, unless this is a grouping
        // turn, where staying on target is the whole point.
        AimTarget target = mainTarget;
        if (!groupingTurn && i > 0 && unitDist(rng_) < 0.20f) {
            target = pickScoringTarget();
        }

        DartPlacement d;
        while (true) {
            d.x = target.x + noiseDist(rng_) + turnBiasX;
            d.y = target.y + noiseDist(rng_) + turnBiasY;
            if (sqrtf(d.x * d.x + d.y * d.y) < maxR) break;
        }
        sampleDartPose(d);
        result.push_back(d);
    }
    return result;
}

void Randomizer::lightPosition(float targetIrr, double boardZ, glm::vec3& pos,
                               float& intensity,
                               float& panDegOut, float& tiltDegOut,
                               float& distOut) {
    // A wide envelope: real rooms put the lamp off to one side, or high behind
    // the thrower, or low from a window, and a model trained on one key-light
    // quadrant is sensitive to lighting in exactly that way. Tilt reaches down
    // to 5 degrees -- near-grazing light, which is what makes the sisal relief
    // and the wire shadows read completely differently -- and pan covers a
    // full half-turn.
    //
    // Distance 6-34 BU. The board is 4.5 BU across, so at 45 BU the
    // inverse-square falloff from one rim to the other is 1.2x -- no visible
    // gradient at all. At 6 BU it is 2.8x, which is what a board lamp on its
    // own bracket actually does. Across 49 real captures, the spread of
    // illumination across same-paint beds is 3-5x larger than a distant lamp
    // can produce.
    std::uniform_real_distribution<float> distDist(6.0f, 34.0f);
    std::uniform_real_distribution<float> panDist(-90.0f, 90.0f);
    std::uniform_real_distribution<float> tiltDist(5.0f, 75.0f);
    // The caller samples the irradiance DELIVERED AT THE BOARD; the lamp's
    // intensity is derived from where it ended up. Drawing intensity directly
    // would make distance an exposure control as well as a gradient control
    // -- irradiance goes as 1/d^2 -- and the near end of the range would blow
    // out frames. Decoupled, distance only sets the gradient across the face.
    float dist = distDist(rng_);
    float pan = glm::radians(panDist(rng_));
    float tilt = glm::radians(tiltDist(rng_));
    // Intensity is set below, once the position is known -- it depends on the
    // angle to the face as well as the distance.

    pos.x = dist * cosf(tilt) * sinf(pan);
    pos.y = dist * sinf(tilt);
    pos.z = dist * cosf(tilt) * cosf(pan);

    // Derive intensity from the irradiance actually DELIVERED to the board
    // face, which depends on the angle the lamp makes with it and not only on
    // how far away it is.
    //
    // Distance alone is not enough. The face normal is +Z and pan reaches
    // +-90 degrees, which puts the lamp level with the board plane: NdotL goes
    // to zero and the direct term vanishes however bright the lamp is, giving
    // a dark tail real captures do not have.
    //
    // This is an auto-exposure emulation, and deliberately: every real capture
    // comes from a camera that pins the scene near a target however the room
    // is lit, which is why their exposure spread is 1.34x while an unregulated
    // renderer's is 1.8x. Grazing light stays available for the contrast it
    // gives without also meaning a dark frame.
    //
    // The NdotL floor caps the correction at ~5x. Without it a lamp exactly in
    // the board's plane would demand unbounded intensity and blow out whatever
    // rim it does reach.
    const glm::vec3 toLight = pos - glm::vec3(0.0f, 0.0f, (float)boardZ);
    const float dCentre = glm::length(toLight);
    const float ndl = glm::max(toLight.z / glm::max(dCentre, 1e-4f), 0.20f);
    intensity = targetIrr * (dCentre * dCentre + 1.0f) / ndl;

    panDegOut = glm::degrees(pan);
    tiltDegOut = glm::degrees(tilt);
    distOut = dist;
}

/// Blackbody colour for a temperature in kelvin, LINEAR RGB.
///
/// Tanner Helland's approximation, which is a fit to display-encoded sRGB
/// values, decoded to linear here. Decoded, 3000K comes out at (1, 0.46,
/// 0.15), within a few percent of the Planckian chromaticity converted to
/// linear sRGB; used undecoded it would pass as roughly 4000K.
///
/// Shared by both fixtures so they sample temperature the same way. Two lights in one room are rarely the same temperature -- a
/// tungsten lamp beside a daylight window, or a warm fitting beside a cool
/// LED -- and that difference across a surface is a strong cue that the light
/// is real.
static glm::vec3 blackbodyRGB(float kelvin) {
    const float t = kelvin / 100.0f;
    float r, g, b;
    if (t <= 66.0f) r = 1.0f;
    else r = std::clamp(1.292936f * powf(t - 60.0f, -0.1332047f), 0.0f, 1.0f);
    if (t <= 66.0f) g = std::clamp(0.390082f * logf(t) - 0.631951f, 0.0f, 1.0f);
    else g = std::clamp(1.129891f * powf(t - 60.0f, -0.0755148f), 0.0f, 1.0f);
    if (t >= 66.0f) b = 1.0f;
    else if (t <= 19.0f) b = 0.0f;
    else b = std::clamp(0.543207f * logf(t - 10.0f) - 1.19625f, 0.0f, 1.0f);
    return srgbToLinear(glm::vec3(r, g, b));
}

FrameState Randomizer::randomize(const RingRadii& radii, double boardZ,
                                 const std::vector<DartGeometry>& variants) {
    FrameState state;

    // The design is chosen FIRST, before anything that depends on the shape.
    // De-intersection, the swept-radius profile and the flight roll all key
    // off it, and the three darts of a turn all take this one.
    shape_ = nullptr;
    const DartGeometry* geom = nullptr;
    if (!variants.empty()) {
        state.dartVariant =
            (int)(std::uniform_int_distribution<size_t>(0, variants.size() - 1)(rng_));
        geom = &variants[state.dartVariant];
        if (dartShapes_.size() != variants.size()) {
            dartShapes_.assign(variants.size(), DartShape{});
            dartShapeGen_.assign(variants.size(), 0);
        }
    }

    // Board rotation
    state.boardRotation = boardRotation();

    // Independent face rotation: multiples of 2 segments (36°) to preserve
    // alternating color pattern while shuffling texture details (logos, wear)
    std::uniform_int_distribution<int> faceDist(0, 9);
    state.boardFaceRotation = faceDist(rng_) * 2.0f * static_cast<float>(SEGMENT_ANGLE_SPAN);

    // Number ring misalignment. Seated by hand, so it is never exactly on a
    // segment boundary. Normal rather than uniform because most boards are
    // close and a badly crooked ring is rare; truncated at 4 degrees so the
    // numerals cannot drift far enough to sit against the wrong bed (half a
    // segment is 9deg).
    {
        const float lim = glm::radians(4.0f);
        state.numberRingOffset =
            truncatedNormal(rng_, 0.0f, glm::radians(1.3f), -lim, lim);
    }

    // Printed marks on the annulus. Most boards carry several, so that is the
    // common case rather than the exception. Each mark is assembled from a
    // random glyph sequence, so no specific mark ever repeats -- see
    // tools/generate_decal_atlas.py for why a fixed set of marks is not
    // enough.
    {
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        std::uniform_int_distribution<int> fontDist(0, kDecalAtlasFontRows - 1);
        std::uniform_int_distribution<int> glyphDist(0, kDecalAtlasGlyphCols - 1);
        // Between the double and the numerals, not the whole annulus: the
        // number ring sits on the annulus, so marks spanning all of it would
        // land underneath the numbers. This is the band real boards print in.
        const float outerR = static_cast<float>(radii.number_ring_inner_r); // 1.95
        const float innerR = static_cast<float>(radii.double_outer_r);      // 1.70
        const float band = (outerR - innerR) * 0.5f;                        // 0.125
        const float mid = (innerR + outerR) * 0.5f;

        float roll = u01(rng_);
        int n = (roll < 0.06f) ? 0 : (roll < 0.22f ? 1 : 2 + (int)(u01(rng_) * 3.0f));
        n = std::min(n, FrameState::kMaxDecals);

        for (int i = 0; i < FrameState::kMaxDecals; ++i) {
            auto& dcl = state.decals[i];
            if (i >= n) { dcl.numGlyphs = 0; dcl.opacity = 0.0f; dcl.halfW = 0.0f; continue; }

            dcl.fontRow = fontDist(rng_);
            dcl.numGlyphs = 2 + (int)(u01(rng_) * (kDecalMaxGlyphs - 1));
            for (int g = 0; g < kDecalMaxGlyphs; ++g) dcl.glyphs[g] = glyphDist(rng_);

            dcl.style = (u01(rng_) < 0.62f) ? 0 : (u01(rng_) < 0.55f ? 1 : 2);
            dcl.shape = (u01(rng_) < 0.5f) ? 0 : 1;
            dcl.stroke = glm::mix(0.05f, 0.13f, u01(rng_));

            // Fill the annulus radially, then run the mark along it. Glyph
            // cells are square, so the quad's width has to be numGlyphs times
            // the glyph height or every letter is stretched.
            dcl.halfH = glm::mix(band * 0.55f, band * 0.95f, u01(rng_));
            float textHalfH = dcl.halfH * ((dcl.style > 0) ? 0.62f : 0.94f) * 0.55f;
            dcl.halfW = textHalfH * (float)dcl.numGlyphs / ((dcl.style > 0) ? 0.62f : 0.94f);

            float th = u01(rng_) * glm::two_pi<float>();
            float slack = band - dcl.halfH;
            float r = mid + glm::mix(-slack, slack, u01(rng_));
            dcl.centre = glm::vec2(r * std::sin(th), r * std::cos(th));
            // local +x tangential, local +y radial: the shader puts local +x
            // along (cos, sin), and the tangent at th is (cos th, -sin th).
            dcl.rotation = -th + glm::mix(-0.04f, 0.04f, u01(rng_));

            // Chord-to-arc sagitta over the mark's half-length, expressed in
            // the quad's own units, so a long mark bends with the rim instead
            // of cutting across it.
            dcl.curvature = (dcl.halfW * dcl.halfW) / (2.0f * r * dcl.halfH);

            // Printing is opaque; only wear fades it.
            dcl.opacity = glm::mix(0.8f, 1.0f, u01(rng_));

            // Board printing is overwhelmingly red or white, with some black
            // and the occasional metallic. Authored in sRGB; the shader blends
            // the tint into linear albedo.
            float pick = u01(rng_);
            glm::vec3 ink;
            if (pick < 0.42f)      ink = glm::vec3(0.74f, 0.11f, 0.13f);
            else if (pick < 0.78f) ink = glm::vec3(0.94f, 0.93f, 0.90f);
            else if (pick < 0.92f) ink = glm::vec3(0.09f, 0.09f, 0.09f);
            else                   ink = glm::vec3(0.78f, 0.66f, 0.38f);
            float g = glm::mix(0.82f, 1.0f, u01(rng_));
            dcl.tint = srgbToLinear(ink) * g;
        }
    }

    // Sisal fibre and wear on the face
    sampleSisal(state, radii);

    // Camera orbit
    state.camera = cameraOrbit(static_cast<float>(radii.board_r), H_APERTURE_MM);

    // Darts
    //
    // Note: placeDarts aims in board-local coordinates (segmentTarget puts
    // segment 20 at +Y) and the result is used as a world coordinate while the
    // board is rotated by boardRotation, so the cluster lands on a segment
    // rotated by a random multiple of 18 degrees from the intended one. That is
    // deliberate rather than a defect to repair: the purpose of the skill model
    // is to produce realistic groupings, and rotation preserves both the radius
    // and the darts' positions relative to each other. Which segment receives
    // the group is then just extra variety.
    state.darts = placeDarts(radii);

    // Darts must not intersect one another. That is not a configuration any
    // real turn can reach -- two solid objects cannot occupy the same space --
    // and it corrupts the labels as well as the image: the
    // instance silhouette has to award contested pixels to one dart, and the
    // fitted boxes of two interpenetrating darts overlap almost entirely.
    //
    // Fixed at the source rather than by rejecting the frame: rejection would
    // bias the dataset away from tight groups, which are the hardest and most
    // valuable case, and the whole point of the skill model is to produce them.
    if (geom != nullptr && !geom->verts.empty()) {
        // Built lazily per slot and rebuilt whenever the slot holds geometry
        // of a different generation -- the pool regenerates slots in place,
        // and a shape from the previous occupant would separate darts that
        // are no longer drawn. Generation 0 is geometry that did not come
        // through extractGeometry and cannot be told apart from an edit, so
        // it is rebuilt every time. Generated geometry is already in board
        // units, so the unit scale here is not a placeholder.
        const uint64_t gen = geom->generation;
        if (gen == 0 || dartShapeGen_[state.dartVariant] != gen) {
            dartShapes_[state.dartVariant] =
                buildShape(geom->bodyVerts, geom->flightVerts, geom->tipLocal,
                           geom->tailLocal, glm::vec3(1.0f));
            dartShapeGen_[state.dartVariant] = gen;
        }
        shape_ = &dartShapes_[state.dartVariant];
    }
    if (shape_ != nullptr) {
        state.maxSeparationBU = separateDarts(
            state.darts, boardZ, (float)radii.board_r * 0.95f,
            state.dartsSeparated);
        // After the bodies are clear, since moving a dart changes where its
        // flight is and would invalidate any roll chosen before the move.
        {
            state.dartsRerolled = resolveFlightRolls(state.darts, boardZ,
                                                     state.flightsUnresolved);
            // Last resort for whatever neither displacement nor lean could
            // clear -- a handful of turns in a thousand, and by this point the
            // overlaps left are a fraction of a millimetre, so the looser floor
            // is spending a few millimetres of ground rather than the tens the
            // full-strength version would have.
            int extra = 0;
            const float more = separateDarts(state.darts, boardZ,
                                             (float)radii.board_r * 0.95f,
                                             extra, 0.12f);
            if (extra > 0) {
                state.dartsSeparated = std::max(state.dartsSeparated, extra);
                state.maxSeparationBU =
                    std::max(state.maxSeparationBU, more);
            }
        }
    }

    state.numDarts = (int)state.darts.size();

    // Build each dart's model matrix from its sampled pose.
    //
    // The dart is posed about its POINT, not about the model's origin.
    // Pivoting on the origin would let the lean carry the point away from the
    // sampled coordinate by (dartZ - boardZ) * tan(lean) -- 11 to 17mm, against
    // a triple bed only 8mm wide, and larger than a pro's aiming error.
    //
    // Pivoting on the point makes the sampled (x, y) exactly where the dart
    // enters the board, which is also what computeAnnotation reports and what
    // computeScoreZone scores.
    state.dartTransforms.resize(state.numDarts);
    state.dartMaterials.resize(state.numDarts);
    // One set of darts per visit.
    //
    // A visit is thrown with one set, so the darts in a frame match, and that
    // is a cue the network can use -- having found one dart, the others look
    // like it. Sampling each dart independently would destroy it.
    //
    // So the set is sampled once and shared, with a little per-dart jitter for
    // wear, plus an occasional genuine odd one out: a replaced flight is
    // common enough that a mismatched dart is a real sight, just not the rule.
    std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);
    auto sampleDartMaterial = [&]() {
        DartMaterial dm;
            // Flight colour, from real flights and converted to LINEAR.
            //
            // A uniform RGB-cube sample used as linear albedo would be wrong
            // twice over: uniform in LINEAR has median 0.5, which displays at
            // sRGB 188, and a cube sample is near-grey far more often than a
            // real flight, which is printed polyester in a handful of strong
            // colours. Across 115 flights in 55 real captures the median luma
            // is 41 and saturation 0.58, and nothing is above luma 150. The
            // flight mesh carries the shaft too and spans half the dart's
            // length, so this is most of what a dart looks like.
            {
                // sRGB bytes, as anyone would read them off a flight.
                static const glm::vec3 kFlights[] = {
                    { 28.0f,  28.0f,  30.0f},   // black
                    { 20.0f,  20.0f,  22.0f},   // black, deeper
                    {198.0f,  32.0f,  38.0f},   // red
                    {176.0f,  26.0f,  30.0f},   // red, darker
                    { 32.0f,  72.0f, 188.0f},   // blue
                    { 24.0f,  54.0f, 140.0f},   // blue, navy
                    { 30.0f, 148.0f,  72.0f},   // green
                    {240.0f, 198.0f,  44.0f},   // yellow
                    {238.0f, 130.0f,  32.0f},   // orange
                    {120.0f,  48.0f, 158.0f},   // purple
                    {226.0f,  78.0f, 146.0f},   // pink
                    {228.0f, 228.0f, 230.0f},   // white
                    {150.0f, 152.0f, 158.0f},   // silver / grey
                };
                const int nf = (int)(sizeof(kFlights) / sizeof(kFlights[0]));
                int fi = glm::min((int)(unitDist(rng_) * (float)nf), nf - 1);
                dm.flightColor = srgbToLinear(kFlights[fi] / 255.0f);
                // Print, wear and the shaft under it all pull the value
                // about; the hue is what stays put.
                dm.flightColor *= glm::mix(0.45f, 0.95f, unitDist(rng_));
                // A flight is almost never one flat colour: it carries a
                // printed pattern, a maker's logo, a white panel. Averaged
                // over the vane that pulls it toward neutral, which is why
                // measured real flights come back less saturated (0.58) than
                // a pure flight colour renders (0.72).
                {
                    const float lum = glm::dot(dm.flightColor,
                                               glm::vec3(0.2126f, 0.7152f, 0.0722f));
                    dm.flightColor = glm::mix(dm.flightColor, glm::vec3(lum),
                                              glm::mix(0.0f, 0.35f, unitDist(rng_)));
                }
            }
            // Flights are laminated or printed polyester and read as genuinely
            // glossy: a sharp specular highlight that moves with the light, not a
            // broad sheen. The upper end covers older or scuffed flights, which do
            // dull with use, but the bulk of the range belongs low.
            dm.flightRoughness = std::uniform_real_distribution<float>(0.04f, 0.35f)(rng_);
        // Roughly a quarter of sets have translucent flights; the rest are
        // solid. A low alpha reads as a missing flight -- the board shows
        // straight through the vane -- and moulded and slim flights diffuse
        // light rather than passing an image, so even the clearest sit well
        // above half.
        dm.flightAlpha = (unitDist(rng_) < 0.28f)
                           ? glm::mix(0.66f, 0.92f, unitDist(rng_))
                           : 1.0f;
            // Barrel finish. Sampled from what darts are actually made and coated
            // as, not from the RGB cube: for a metal the base colour is its F0
            // reflectance, which is bright and specific, so uniform cube samples
            // produce dark muddy metals that do not exist. Coloured barrels are
            // real and common enough to keep a decent share -- anodised and
            // titanium-nitride finishes come in most colours -- but bare tungsten
            // and steel are still the bulk of what is in use.
            //
            // Roughness ranges are deliberately well clear of mirror. Metals
            // receive no diffuse term, so reflection is their ENTIRE response
            // and roughness decides what a barrel looks like rather than
            // merely tinting it. Materials are per-set, so one shiny draw
            // makes every dart in the frame shiny.
            //
            // Grip texture is the justification for the absolute values. A
            // real barrel is knurled, ringed or shark-cut over most of its
            // length at roughly 0.5mm pitch -- far below a pixel at the
            // framings this renders, where a whole barrel spans about 12. Sub
            // pixel geometry of that kind IS roughness; modelling it as a
            // texture would alias, so raising roughness is not a stand-in for
            // the real thing here, it is the correct representation of it.
            {
                float pick = unitDist(rng_);
                auto jitter = [&](glm::vec3 c, float amt) {
                    return glm::clamp(c * (1.0f - amt + 2.0f * amt * unitDist(rng_)),
                                      glm::vec3(0.0f), glm::vec3(1.0f));
                };
                if (pick < 0.38f) {
                    // Bare tungsten: dark gunmetal, knurled so not smooth.
                    //
                    // Below the ~0.50 that polished pure tungsten measures.
                    // A barrel is a sintered 80-95% tungsten alloy that has
                    // been bead-blasted and cut, and a blasted sintered surface
                    // loses a lot of light to multiple scattering between
                    // microfacets -- which this shader's single-scatter GGX
                    // does not model, so the loss has to come out of F0
                    // instead.
                    dm.metalColor = jitter(glm::vec3(0.45f, 0.44f, 0.42f), 0.22f);
                    dm.metalRoughness = glm::mix(0.42f, 0.72f, unitDist(rng_));
                    dm.metalMetallic = 1.0f;
                } else if (pick < 0.58f) {
                    // Polished steel or nickel.
                    //
                    // 0.60 sits between chrome (0.55) and nickel (0.66), which
                    // is what these barrels actually are. Brighter (~0.78) is
                    // aluminium or silver -- no dart is made of either, and
                    // against the HDR environment it drives the whole barrel
                    // to white rather than giving it a bright highlight on a
                    // grey body.
                    dm.metalColor = jitter(glm::vec3(0.66f, 0.66f, 0.68f), 0.13f);
                    // Satin-polished, not mirror. A true mirror barrel exists but
                    // is rare.
                    dm.metalRoughness = glm::mix(0.26f, 0.50f, unitDist(rng_));
                    dm.metalMetallic = 1.0f;
                } else if (pick < 0.74f) {
                    // Black coating. Still a coating over metal, so metallic stays
                    // high, but the reflectance is very low.
                    // This matches the darkest bin of real captures to within a
                    // tenth of a percent.
                    dm.metalColor = jitter(glm::vec3(0.09f, 0.09f, 0.10f), 0.35f);
                    dm.metalRoughness = glm::mix(0.40f, 0.72f, unitDist(rng_));
                    dm.metalMetallic = glm::mix(0.7f, 1.0f, unitDist(rng_));
                } else {
                    // Anodised or titanium-nitride.
                    //
                    // Drawn from the finishes that are actually sold rather than
                    // from a random hue: a cube sample is near-grey far more
                    // often than a real anodised finish, and coloured barrels on
                    // real darts measure a median saturation of about 0.31.
                    static const glm::vec3 kAnodised[] = {
                        {0.76f, 0.58f, 0.22f},   // titanium nitride gold
                        {0.18f, 0.29f, 0.56f},   // blue
                        {0.56f, 0.15f, 0.14f},   // red
                        {0.16f, 0.42f, 0.24f},   // green
                        {0.35f, 0.20f, 0.48f},   // purple
                        {0.24f, 0.23f, 0.26f},   // graphite / gunmetal
                        {0.62f, 0.44f, 0.18f},   // bronze
                    };
                    const int n = (int)(sizeof(kAnodised) / sizeof(kAnodised[0]));
                    int idx = glm::min((int)(unitDist(rng_) * (float)n), n - 1);
                    // Same 0.7 scaling as the neutral finishes below their
                    // textbook F0, for the same reason: these are coatings on
                    // a blasted barrel, not polished slabs of the metal.
                    dm.metalColor = jitter(kAnodised[idx], 0.20f);
                    dm.metalRoughness = glm::mix(0.30f, 0.58f, unitDist(rng_));
                    dm.metalMetallic = 1.0f;
                }
            }

            // Point: steel, duller and darker than the barrel it is fitted to, and
            // sometimes blackened.
            {
                bool blackened = unitDist(rng_) < 0.3f;
                if (blackened) {
                    dm.pointColor = glm::vec3(glm::mix(0.05f, 0.14f, unitDist(rng_)));
                    dm.pointRoughness = glm::mix(0.35f, 0.7f, unitDist(rng_));
                } else {
                    float v = glm::mix(0.45f, 0.68f, unitDist(rng_));
                    dm.pointColor = glm::vec3(v, v, v * 1.02f);
                    dm.pointRoughness = glm::mix(0.25f, 0.55f, unitDist(rng_));
                }
                dm.pointMetallic = 1.0f;
            }
        return dm;
    };

    // The set's finish comes from the VARIANT, not from an independent draw.
    // A brass barrel is fat because brass is half tungsten's density, and it
    // is also yellow; sampling the colour separately would put a tungsten
    // finish on a brass-sized barrel and quietly cancel the one axis the
    // generator exists to introduce. sampleDartMaterial is used only for the
    // odd-one-out replaced flight below, and as the fallback when there is no
    // pool at all.
    DartMaterial setMaterial = sampleDartMaterial();
    if (geom != nullptr) {
        const DartFinish& f = geom->finish;
        setMaterial.flightColor = f.flightColor;
        setMaterial.flightRoughness = f.flightRoughness;
        setMaterial.flightAlpha = f.flightAlpha;
        setMaterial.metalColor = f.metalColor;
        setMaterial.metalMetallic = 1.0f;
        setMaterial.metalRoughness = f.metalRoughness;
        setMaterial.pointColor = f.pointColor;
        setMaterial.pointMetallic = 1.0f;
        setMaterial.pointRoughness = f.pointRoughness;
    }

    for (int i = 0; i < state.numDarts; ++i) {
        auto& d = state.darts[i];
        // Unit scale. Generated darts are built directly in board units, so
        // there is no authored size to recover.
        const glm::vec3 scale(1.0f);

        // Dart axis, pointing from the point toward the flight, i.e. out of the
        // board. Local +Z maps onto this.
        glm::vec3 axis(std::sin(d.tilt) * std::cos(d.azimuth),
                       std::sin(d.tilt) * std::sin(d.azimuth),
                       std::cos(d.tilt));

        // Any orthonormal frame with `axis` as its third column will do; the
        // roll about it is then set explicitly, so the reference vector only
        // has to avoid being parallel to the axis.
        glm::vec3 ref = (std::abs(axis.z) < 0.999f) ? glm::vec3(0.0f, 0.0f, 1.0f)
                                                    : glm::vec3(1.0f, 0.0f, 0.0f);
        glm::vec3 tangent = glm::normalize(glm::cross(ref, axis));
        glm::mat3 frame(tangent, glm::cross(axis, tangent), axis);

        float cr = std::cos(d.roll), sr = std::sin(d.roll);
        glm::mat3 rollZ(glm::vec3(cr, sr, 0.0f),
                        glm::vec3(-sr, cr, 0.0f),
                        glm::vec3(0.0f, 0.0f, 1.0f));
        glm::mat3 rot = frame * rollZ;

        // Where the point tip has to end up: along the axis, `penetration`
        // below the entry point on the board face.
        glm::vec3 entry(d.x, d.y, (float)boardZ);
        glm::vec3 tipWorld = entry - axis * d.penetration;

        glm::mat3 rotScale = rot * glm::mat3(glm::vec3(scale.x, 0.0f, 0.0f),
                                             glm::vec3(0.0f, scale.y, 0.0f),
                                             glm::vec3(0.0f, 0.0f, scale.z));
        // Generated darts have their tip at the origin, so this is a no-op for
        // them -- kept because the pose maths should not silently depend on
        // that, and a future variant with a different anchor would break
        // quietly otherwise.
        const glm::vec3 tipLocal = geom ? geom->tipLocal : glm::vec3(0.0f);
        glm::vec3 origin = tipWorld - rotScale * tipLocal;

        glm::mat4 M(rotScale);
        M[3] = glm::vec4(origin, 1.0f);
        state.dartTransforms[i] = M;

        // Same set, lightly worn differently.
        DartMaterial& dm = state.dartMaterials[i];
        if (unitDist(rng_) < 0.10f) {
            // Odd one out: a replaced flight, which people do play with.
            DartMaterial other = sampleDartMaterial();
            dm = setMaterial;
            dm.flightColor = other.flightColor;
            dm.flightRoughness = other.flightRoughness;
        } else {
            dm = setMaterial;
            // Uniform jitter, restricted to the part of the window inside the
            // bounds rather than clamped onto them. The flight's upper bound
            // clears the variant sampler's 0.58, so a matte flight keeps its
            // spread instead of collapsing to one value.
            auto wear = [&](float v, float amt, float lo, float hi) {
                return truncatedUniform(rng_, v, amt, lo, hi);
            };
            dm.flightRoughness = wear(dm.flightRoughness, 0.04f, 0.02f, 0.62f);
            dm.metalRoughness  = wear(dm.metalRoughness,  0.05f, 0.05f, 0.7f);
            dm.pointRoughness  = wear(dm.pointRoughness,  0.05f, 0.15f, 0.8f);
        }
    }

    // Light
    // The frame's overall light level, drawn ONCE and shared by every fixture
    // and by the ambient fill. Log-uniform because exposure is perceived
    // multiplicatively.
    //
    // One number, because the alternative compounds: with the lamp's strength
    // and the fill/direct balance drawn independently, a dim lamp could land
    // in a fill-heavy frame and multiply into a board far darker than a camera
    // would ever deliver. Real captures have a tight exposure spread
    // precisely because the phone auto-exposes, and this renderer has no such
    // loop.
    std::uniform_real_distribution<float> frameIrrU(0.0f, 1.0f);
    const float frameIrr = expf(glm::mix(logf(1.38f), logf(2.62f),
                                         frameIrrU(rng_)));

    float keyPanDeg = 0.0f, keyTiltDeg = 0.0f, keyDist = 0.0f;
    lightPosition(frameIrr, boardZ, state.lightPos, state.lightIntensity,
                  keyPanDeg, keyTiltDeg, keyDist);

    // Half the frames use a ring fixture mounted around the board instead of a
    // distant point light. Its defining property is that it lights the face
    // head-on from all sides at once, so the radial components of the incoming
    // light largely cancel and shadows are weak and multi-directional rather
    // than one hard cast. The diffuse term is approximated by placing the
    // light on the board axis, which reproduces that frontal character; the
    // shadow rays sample the actual ring, which is what makes the shadows
    // behave like a ring's rather than a point's.
    {
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        // 32%: a ring fixture is deliberately near-shadowless, so it should
        // not dominate the training set. Ring-lit boards are real and stay
        // well represented; they are just not the default.
        state.lightRingMode = (u01(rng_) < 0.32f) ? 1 : 0;
        state.lightRingRadius = 2.35f + 0.65f * u01(rng_);   // just outside the board
        state.lightRingZ      = 0.5f + 0.9f * u01(rng_);     // stand-off from the face
        if (state.lightRingMode == 1) {
            // lightPos.xyz is unused in ring mode -- the shader integrates over
            // the ring itself -- but lightPos.w still carries intensity. Kept
            // on-axis so anything reading the position gets something sane.
            //
            // Radius and stand-off are sampled wide on purpose: with the ring
            // integrated properly they change the image, and a narrow band
            // would make all ring frames look alike.
            state.lightPos = glm::vec3(0.0f, 0.0f, state.lightRingZ + 1.2f);
            // Same delivered-irradiance basis as the point path, derived from
            // the ring's own geometry, so both modes share one exposure scale
            // and changing either cannot silently unbalance them.
            const float ringAboveFace = state.lightRingZ - (float)boardZ;
            const float repD2 = state.lightRingRadius * state.lightRingRadius
                              + ringAboveFace * ringAboveFace;
            state.lightIntensity = frameIrr * (repD2 + 0.02f);
        }

        // Ring emission profile.
        //
        // A perfect uniform ring lights a board perfectly evenly -- and that
        // evenness is GEOMETRIC, so no lamp-distance or ambient change can
        // remove it. Real fixtures are not circles:
        // many are horseshoes with a break at the mount or the cable entry,
        // and an LED ring runs hot and dull around its circumference, often
        // with a dead segment.
        //
        // Gap on 45% of ring frames, up to ~50 degrees of arc. Asymmetry
        // always, since no fixture is perfectly even. The shader normalises by
        // total emission, so these redistribute the fixture's output rather
        // than dimming it -- a fixture is specified by total lumens, and what
        // a horseshoe changes is where they go.
        const float gapCentre = u01(rng_) * 6.2831853f;
        const float gapHalf = (u01(rng_) < 0.45f)
                                ? glm::mix(0.12f, 0.9f, u01(rng_)) : 0.0f;
        const float asymAmp = glm::mix(0.05f, 0.55f, u01(rng_));
        const float asymPhase = u01(rng_) * 6.2831853f;
        state.lightRingProfile = glm::vec4(gapCentre, gapHalf, asymAmp,
                                           asymPhase);
    }

    // Key light colour: temperature from warm (3000K) to cool (7000K).
    state.lightColor = blackbodyRGB(
        std::uniform_real_distribution<float>(3000.0f, 7000.0f)(rng_));

    // Ambient is a scale on the environment probe's radiance rather than a
    // flat fill. A mid-grey room decodes to ~0.216 linear, so 1.0 gives ~0.22
    // for an average environment while letting a dark room stay dark and a
    // bright one lift the shadows.
    //
    // Left at 1.0 here ON PURPOSE. This is an intermediate default: the real
    // ambient is drawn at the END of randomize(), anti-correlated with the
    // direct intensity. Do not add a random draw here; that block overwrites
    // it, so the only effect would be to shift the RNG stream.
    state.ambient           = 1.0f;
    // Kept low deliberately: a normal map derived from a photograph rather
    // than measured amplifies denoise artifacts into relief that is not on the
    // real board at full strength.
    state.normalStrength    = 0.45f;
    state.envIntensity      = 0.8f;
    state.contactAOStrength = 1.0f;
    state.shadowStrength    = 0.95f;

    // Ceiling fixture. Radiance runs from a dim domestic room to a bright hall
    // -- the point of the range is the RATIO against the photo probe, which is
    // capped at 1.0, so even the low end is several times the wall behind the
    // board. Colour spans warm tungsten to cool fluorescent, the two things
    // venues actually have; a neutral white fixture is the rarer case.
    //
    // The two endpoints are the usual display swatches for a tungsten lamp
    // and a cool-white tube, so they are decoded and mixed in linear.
    {
        const float warm = unitDist(rng_);
        state.envRoomColor = glm::mix(
            srgbToLinear(glm::vec3(1.00f, 0.78f, 0.55f)),   // tungsten
            srgbToLinear(glm::vec3(0.82f, 0.90f, 1.00f)),   // fluorescent
            warm);
        state.envRoomRadiance = glm::mix(2.0f, 22.0f, unitDist(rng_))
                                * envRoomScale_;
    }

    // Number font variant
    if (NUM_FONT_VARIANTS > 0) {
        std::uniform_int_distribution<int> fontDist(0, NUM_FONT_VARIANTS - 1);
        state.fontVariant = fontDist(rng_);
    }

    // Number scale
    std::uniform_int_distribution<int> scaleDist(0, (int)NUMBER_SCALES.size() - 1);
    state.numberScale = NUMBER_SCALES[scaleDist(rng_)];

    // Number material: 50/50 metallic vs plastic
    state.numberMetallic = unitDist(rng_) >= 0.5f;

    // Background Perlin noise. Colours are drawn uniform over DISPLAY values
    // -- what the colour looks like -- and decoded, since the background pass
    // writes them straight into an sRGB target. Uniform in linear would put
    // the median at a display value of 188 and leave dark backgrounds rare.
    state.bgColorA = srgbToLinear(
        glm::vec3(unitDist(rng_), unitDist(rng_), unitDist(rng_)));
    state.bgColorB = srgbToLinear(
        glm::vec3(unitDist(rng_), unitDist(rng_), unitDist(rng_)));
    std::uniform_real_distribution<float> scaleFDist(4.0f, 16.0f);
    std::uniform_real_distribution<float> offsetDist(0.0f, 100.0f);
    std::uniform_real_distribution<float> intensityDist(0.3f, 1.0f);
    state.bgNoiseScale = scaleFDist(rng_);
    state.bgNoiseOffsetX = offsetDist(rng_);
    state.bgNoiseOffsetY = offsetDist(rng_);
    state.bgIntensity = intensityDist(rng_);

    // Spider finish and profile.
    //
    // Placed late in randomize() on purpose: every draw from rng_ shifts the
    // ones after it, so new draws belong at the end if the quantities before
    // them are to stay reproducible for a given seed.
    {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        const float pick = u(rng_);
        if (pick < 0.42f) {
            // Bright plated or stainless staple. Reads as a specular line
            // against the bed, which is what makes wires usable landmarks.
            const float g = glm::mix(0.80f, 0.95f, u(rng_));
            state.wireColor = glm::vec3(g, g * glm::mix(0.985f, 1.0f, u(rng_)),
                                        g * glm::mix(0.95f, 1.0f, u(rng_)));
            state.wireRoughness = glm::mix(0.12f, 0.28f, u(rng_));
            state.wireMetallic  = 1.0f;
        } else if (pick < 0.76f) {
            // Dulled, oxidised or simply old. Still bare metal, so metallic
            // stays at one; it is the scattering that changes.
            const float g = glm::mix(0.42f, 0.70f, u(rng_));
            state.wireColor = glm::vec3(g, g * glm::mix(0.97f, 1.0f, u(rng_)),
                                        g * glm::mix(0.92f, 0.99f, u(rng_)));
            state.wireRoughness = glm::mix(0.30f, 0.55f, u(rng_));
            state.wireMetallic  = 1.0f;
        } else {
            // Black powder coating, standard on most modern boards. The coat is
            // a dielectric film over the wire, so metallic drops well away from
            // one -- treating it as bare dark metal gives a black mirror, which
            // is not what a coated spider looks like.
            const float g = glm::mix(0.025f, 0.085f, u(rng_));
            state.wireColor = glm::vec3(g, g, g * glm::mix(1.0f, 1.15f, u(rng_)));
            state.wireRoughness = glm::mix(0.30f, 0.60f, u(rng_));
            state.wireMetallic  = glm::mix(0.05f, 0.25f, u(rng_));
        }

        if (kSpiderVariants > 0) {
            std::uniform_int_distribution<int> vd(0, kSpiderVariants - 1);
            state.spiderVariant = vd(rng_);
        }
    }

    // Ambient fill against direct light.
    //
    // Fill and direct are drawn anti-correlated from one variable, so a frame
    // sits somewhere on a flat-to-contrasty axis: a bright diffuse room with
    // soft shadows at one end, a single hard lamp in a dark room at the other.
    // That is the actual variation between the places a board gets
    // photographed. Scaling the direct range alone cannot balance exposure,
    // because it moves the blown and dark tails the same way.
    //
    // Deliberately spread rather than tuned onto one target: the real
    // reference captures come from one room, and matching them exactly would
    // fit the renderer to a single environment, which is the opposite of what
    // the randomisation is for. Some frames should be over-exposed and some
    // under, because real captures are.
    //
    // Note what this does NOT change. Lighting falloff within a single bed
    // colour matches real photographs closely -- cream p90/p10 of 1.48 against
    // 1.47, black 3.18 against 3.38. The face's wider span between median and
    // p99 is the cream-to-black albedo separation, which no lighting change
    // reaches, so the blown and dark tails cannot both be closed from here.
    {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        const float t = u(rng_);
        // The hard-light end matters: a dart's shadow is one of the few
        // direct cues to how far it stands off the surface, which is exactly
        // what the tip/flight separation depends on, and too much fill washes
        // it out. A room with a dedicated board lamp throws hard shadows.
        //
        // The bottom end (0.15) is the lever that decides whether a frame has
        // directional light at all. The environment is DIRECTION-ONLY, so on
        // a flat board every point sees the same hemisphere and
        // `ambient * irradiance` lands as a spatially uniform wash; only the
        // positioned light varies across the face. With the HDR ceiling
        // fixture able to drive irradiance past 2, a higher floor lets the
        // uniform term beat the directional one in most frames.
        //
        // The top end stays high: a bright evenly-lit room is a real
        // condition.
        //
        // Scaled by the frame's own light level, so `t` sets the fill-to-direct
        // RATIO and not the exposure. Unscaled, a dim frame would keep a full-
        // strength fill and a bright one a weak one, widening the exposure
        // spread instead of trading contrast against fill.
        //
        // 1.90 is the geometric mean of the frameIrr range, so the average
        // frame is unchanged by the scaling.
        state.ambient = glm::mix(0.15f, 3.2f, t) * (frameIrr / 1.90f);
        // Direct falls as fill rises, holding overall exposure roughly put
        // while the spatial contrast the direct term creates comes down. The
        // top end compensates for the lower fill, so the lit side keeps its
        // exposure while the shadow side actually goes dark.
        //
        // Balanced against the RT path, which traces sixteen ambient rays and
        // multiplies by contact AO, so its ambient delivers materially less
        // than the non-RT path's single env lookup along the normal. A lower
        // direct floor leaves the fill-heavy end dark on the RT path.
        state.lightIntensity *= glm::mix(2.05f, 1.05f, t);

        // Second fixture.
        //
        // Sampled AFTER the ambient/direct axis so it scales with the key
        // light: a frame chosen to be hard-lit should not have its shadows
        // filled back in by a second lamp at full strength.
        //
        // This is what addresses flat lighting structurally. The environment
        // probe carries a ceiling fixture, but a probe is
        // DIRECTION-ONLY -- on a flat board every point sees the same
        // hemisphere, so its contribution is spatially uniform and cannot
        // produce a gradient across the face no matter how bright it is. Only
        // a positioned light can. One positioned light gives one gradient and
        // one shadow direction; real rooms overlap two, and the soft
        // interference between them is most of what reads as real.
        if (u(rng_) < 0.62f) {
            // Placed RELATIVE to the key, not drawn independently.
            //
            // Independent placement would put the two fixtures on opposite
            // sides of the board as often as not, and two opposed lights
            // cancel each other's gradient: measured by illumination spread
            // across same-paint beds, such two-light frames come out FLATTER
            // than one-light frames (median 0.108 against 0.142).
            //
            // Two fittings in a room are usually both overhead and separated
            // horizontally -- so a similar tilt, a substantial pan offset, and
            // a comparable distance. That reinforces the vertical gradient
            // instead of erasing it, and is also just what rooms look like.
            const float side = (u(rng_) < 0.5f) ? -1.0f : 1.0f;
            const float pan2 = glm::radians(keyPanDeg
                                            + side * glm::mix(40.0f, 145.0f, u(rng_)));
            // Within 18 degrees of the key, sampled over the part of that
            // window inside 5-80 rather than clamped onto its edges.
            const float tilt2 = glm::radians(
                truncatedUniform(rng_, keyTiltDeg, 18.0f, 5.0f, 80.0f));
            const float dist2 = keyDist * glm::mix(0.7f, 1.5f, u(rng_));
            glm::vec3 p2(dist2 * cosf(tilt2) * sinf(pan2),
                         dist2 * sinf(tilt2),
                         dist2 * cosf(tilt2) * cosf(pan2));
            // Scaled to its own distance so the fill ratio below means what it
            // says: a fixture twice as far delivers a quarter the irradiance.
            const float i2 = state.lightIntensity * (dist2 * dist2 + 1.0f)
                             / (keyDist * keyDist + 1.0f);
            state.light2Pos = p2;
            // A fill, not a rival. Comparable-strength pairs happen (two
            // ceiling fittings) but a dominant key with a weaker second is
            // the common room, and an even pair cancels both gradients.
            // SPLIT of the frame's light budget, not an addition to it. A
            // second fitting that simply added its own output would make
            // two-light rooms brighter than one-light rooms by up to 65%, an
            // exposure swing the camera would have corrected for and the
            // renderer has no auto-exposure to correct with.
            const float fill = glm::mix(0.12f, 0.65f, u(rng_));
            state.light2Intensity = i2 * fill / (1.0f + fill);
            state.lightIntensity /= (1.0f + fill);
            // Independent temperature, deliberately: two fittings in one room
            // are usually NOT the same, and cross-temperature lighting is a
            // strong naturalness cue. Widened past the key's 3000-7000K at
            // the cool end to cover daylight through a window.
            state.light2Color = blackbodyRGB(glm::mix(2700.0f, 8000.0f,
                                                      u(rng_)));
        } else {
            // Single-fixture rooms are real and stay common.
            state.light2Intensity = 0.0f;
        }
    }

    return state;
}

} // namespace dart
