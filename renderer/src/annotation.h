#pragma once

#include "constants.h"
#include "randomizer.h"

#include "dart_mesh.h"

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace dart {

struct KeypointProjection {
    std::string name;
    float x, y;
};

struct DartAnnotation {
    float x, y;          // projected 2D landing point (dart axis ∩ board plane)
    float flightX, flightY;  // projected 2D rear tip of the flight
    bool flightInFront;      // flight tip is in front of the camera
    // The oriented box, as four projected corners in order:
    // tip-left, flight-left, flight-right, tip-right, so edge 0->1 runs from
    // the tip end toward the flight end. Points rather than a
    // parameterisation so they ride augmentation, and projected from the
    // dart's 3D bounds rather than inferred from tip/flight, which cannot
    // account for perspective turning across-axis extent into along-axis.
    float boxX[4], boxY[4];
    // The dart projects end-on: tip and flight land on the same pixel, so
    // the box is a minimum-area fit whose axis does not come from the dart.
    // Its corners still follow the order above as closely as the projected
    // axis allows, but its direction and which end is "tip" carry no
    // information; consumers should not supervise orientation from it.
    bool boxEndOn;
    // Score of the bed the landing point is in: "DB", "SB", "S20", "T20",
    // "D20", ..., or "MISS" outside the double ring.
    std::string scoreZone;
};

struct FrameAnnotation {
    int frameId;
    std::string imagePath;
    std::vector<KeypointProjection> boardKeypoints;
    std::vector<DartAnnotation> darts;
    float focalLength;
    int width, height;
    int numDarts;
    float boardRotationDeg;
    float boardFaceRotationDeg;

    /// How the frame was lit. Emitted because it is not recoverable from the
    /// image, and lighting statistics (e.g. ring-lit vs point-lit frames, one
    /// vs two fixtures) need to be split by it.
    int lightRingMode;   ///< 1 = ring fixture around the board, 0 = point
    int numLights;       ///< 1 or 2 positioned fixtures
    float ambientLevel;

    /// Camera pose and clip planes, needed to turn the segmentation pass's
    /// window-space depth into a height above the board plane. Without these
    /// the depth buffer is uninterpretable outside the renderer. The
    /// projection itself is cameraProjection (board_transforms.h) on
    /// focalLength, width and height.
    glm::mat4 viewMatrix = glm::mat4(1.0f);
    float nearPlane = kNearPlane;
    float farPlane  = kFarPlane;
};

// Compute all annotations for a frame.
// geom: the turn's dart design. One, not three -- a matched set.
// When null, the flight point falls back to the landing point and is marked
// as not in front of the camera.
FrameAnnotation computeAnnotation(
    int frameId,
    const FrameState& state,
    const RingRadii& radii,
    double boardZ,
    float hAperture,
    int width, int height,
    const DartGeometry* geom = nullptr);

// True when both of every dart's points — the landing point and the flight tip
// — are in front of the camera and project inside the frame.  Occlusion is
// deliberately not considered: a point hidden behind another dart still counts
// as present, since its position is still well defined and inferable from the
// visible geometry.
bool allDartPointsInFrame(const FrameAnnotation& ann, int width, int height);

// Draw frame states until every dart has both points inside the frame, so
// generated image/annotation pairs always carry the full pair.  Annotations are
// pure CPU and computed before any GPU work, so rejected draws are nearly free.
// Falls back to the last draw after maxAttempts.
FrameState randomizeWithDartsInFrame(
    Randomizer& randomizer,
    int frameId,
    const RingRadii& radii,
    double boardZ,
    float hAperture,
    int width, int height,
    const std::vector<DartGeometry>& dartVariants,
    FrameAnnotation& outAnnotation,
    int maxAttempts = 16);

// The annotation as JSON: the one serialisation of FrameAnnotation. The
// dartboard_gen CLI writes it to disk and the Python binding converts it to a
// dict, so both carry exactly the same keys.
nlohmann::json annotationToJson(const FrameAnnotation& ann);

// ---------------------------------------------------------------------------
// Internal helpers (exposed for testing)
// ---------------------------------------------------------------------------

// Compute the NUM_KEYPOINTS board keypoints in 3D world space, named and
// ordered as make_keypoint_names()
std::vector<std::pair<std::string, glm::vec3>> getBoardKeypoints3D(
    const RingRadii& radii, double boardZ, float rotationAngle);

// Project 3D points to 2D pixel coordinates
std::vector<glm::vec2> projectPoints3D(
    const std::vector<glm::vec3>& points,
    const glm::mat4& viewMatrix,
    const glm::mat4& projMatrix,
    int width, int height);

// Score zone ("DB", "SB", "S20", "T20", "D20", "MISS", ...) of a board-local
// XY
std::string computeScoreZone(float x, float y, const RingRadii& radii);

} // namespace dart
