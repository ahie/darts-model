#include "annotation.h"

#include "board_transforms.h"

#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <cmath>
#include <algorithm>
#include <cfloat>

namespace dart {

// ---------------------------------------------------------------------------
// 3D board keypoints
// ---------------------------------------------------------------------------

std::vector<std::pair<std::string, glm::vec3>> getBoardKeypoints3D(
    const RingRadii& radii, double boardZ, float rotationAngle) {

    static const std::array<std::string, NUM_KEYPOINTS> kNames =
        make_keypoint_names();

    const float cosRot = cosf(rotationAngle);
    const float sinRot = sinf(rotationAngle);
    const float z = (float)boardZ;
    // A point at radius R, `halfSegments` half-segments clockwise from
    // segment 20's centre, turned by the board rotation.
    auto at = [&](double R, float halfSegments) {
        const float theta = halfSegments * 0.5f * (float)SEGMENT_ANGLE_SPAN;
        const float x = (float)R * sinf(theta);
        const float y = (float)R * cosf(theta);
        return glm::vec3(x * cosRot - y * sinRot, x * sinRot + y * cosRot, z);
    };

    // Positions in the index order documented in constants.h. The segment
    // keypoints sit at a segment's angular CENTRE; the crossings sit half a
    // segment round, on the wire between segment i and segment i+1, where it
    // meets the ring.
    std::array<glm::vec3, NUM_KEYPOINTS> pos;
    pos[0] = glm::vec3(0.0f, 0.0f, z);
    for (int i = 0; i < 20; ++i) {
        pos[1 + i]  = at(radii.double_outer_r, 2.0f * i);
        pos[21 + i] = at(radii.triple_inner_r, 2.0f * i);
        pos[41 + i] = at(radii.double_outer_r, 2.0f * i + 1.0f);
        pos[61 + i] = at(radii.double_inner_r, 2.0f * i + 1.0f);
    }

    std::vector<std::pair<std::string, glm::vec3>> keypoints;
    keypoints.reserve(NUM_KEYPOINTS);
    for (int k = 0; k < NUM_KEYPOINTS; ++k)
        keypoints.emplace_back(kNames[k], pos[k]);
    return keypoints;
}

// ---------------------------------------------------------------------------
// 3D to 2D projection
// ---------------------------------------------------------------------------

std::vector<glm::vec2> projectPoints3D(
    const std::vector<glm::vec3>& points,
    const glm::mat4& viewMatrix,
    const glm::mat4& projMatrix,
    int width, int height) {

    glm::mat4 vp = projMatrix * viewMatrix;
    std::vector<glm::vec2> result;
    result.reserve(points.size());

    for (auto& pt : points) {
        glm::vec4 clip = vp * glm::vec4(pt, 1.0f);
        glm::vec3 ndc = glm::vec3(clip) / clip.w;

        // NDC to pixel: x_px = (ndc_x + 1) / 2 * W, y_px = (1 - ndc_y) / 2 * H
        float px = (ndc.x + 1.0f) * 0.5f * width;
        float py = (1.0f - ndc.y) * 0.5f * height;

        result.emplace_back(px, py);
    }

    return result;
}

// ---------------------------------------------------------------------------
// Score zone computation
// ---------------------------------------------------------------------------

std::string computeScoreZone(float x, float y, const RingRadii& radii) {

    float r = sqrtf(x * x + y * y);

    // theta: +X = 0, counter-clockwise positive (standard atan2)
    float theta = fmodf(atan2f(y, x) + 2.0f * (float)M_PI, 2.0f * (float)M_PI);

    if (r > (float)radii.double_outer_r)
        return "MISS";
    if (r <= (float)radii.inner_bull_r)
        return "DB";
    if (r <= (float)radii.outer_bull_r)
        return "SB";

    // Convert to board convention (theta=0 at +Y / segment 20, clockwise)
    float thetaBoard = fmodf((float)M_PI / 2.0f - theta + 2.0f * (float)M_PI,
                             2.0f * (float)M_PI);

    float halfSpan = (float)SEGMENT_ANGLE_SPAN / 2.0f;
    float adjusted = fmodf(thetaBoard + halfSpan + 2.0f * (float)M_PI,
                           2.0f * (float)M_PI);
    int segIdx = std::min((int)(adjusted / (float)SEGMENT_ANGLE_SPAN), 19);
    int segNum = SEGMENT_ORDER[segIdx];

    std::string prefix;
    if (r <= (float)radii.triple_inner_r)
        prefix = "S";
    else if (r <= (float)radii.triple_outer_r)
        prefix = "T";
    else if (r <= (float)radii.double_inner_r)
        prefix = "S";
    else
        prefix = "D";

    return prefix + std::to_string(segNum);
}

// ---------------------------------------------------------------------------
// Minimum-area oriented box of a projected point set
// ---------------------------------------------------------------------------

/// Convex hull, monotone chain, counter-clockwise, no collinear points.
static std::vector<glm::vec2> convexHull2D(std::vector<glm::vec2> pts) {
    if (pts.size() < 3) return pts;
    std::sort(pts.begin(), pts.end(), [](const glm::vec2& a, const glm::vec2& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    pts.erase(std::unique(pts.begin(), pts.end(),
                          [](const glm::vec2& a, const glm::vec2& b) {
                              return a.x == b.x && a.y == b.y;
                          }), pts.end());
    if (pts.size() < 3) return pts;

    auto cross = [](const glm::vec2& o, const glm::vec2& a, const glm::vec2& b) {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    };
    std::vector<glm::vec2> hull(2 * pts.size());
    size_t k = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) --k;
        hull[k++] = pts[i];
    }
    for (size_t i = pts.size() - 1, t = k + 1; i > 0; --i) {
        while (k >= t && cross(hull[k - 2], hull[k - 1], pts[i - 1]) <= 0) --k;
        hull[k++] = pts[i - 1];
    }
    hull.resize(k ? k - 1 : 0);
    return hull;
}

/// Tight oriented box with its long axis along `u`, from the projected points.
///
/// The orientation is GIVEN, not minimised. Minimum area is a different
/// objective and the wrong one here: a dart's silhouette is hammer-shaped --
/// thin shaft, wide flight at one end -- so for a low aspect ratio a diagonal
/// box genuinely wins on area. A min-area fit puts the box's long axis 11.6
/// degrees off the dart at the median, over 10 degrees for 61% of darts, and
/// up to 43 degrees on foreshortened ones, where the shaft is shortest
/// relative to the flight.
///
/// The box exists so a query can sample ALONG the dart, so the dart's own axis
/// is the orientation that matters. Extents come from the real projected
/// vertices, which is what makes it tight in that frame.
static void orientedBoxAlong(const std::vector<glm::vec2>& pts, glm::vec2 u,
                             float* outX, float* outY) {
    if (pts.empty()) {
        for (int i = 0; i < 4; ++i) { outX[i] = 0.0f; outY[i] = 0.0f; }
        return;
    }
    glm::vec2 v(-u.y, u.x);
    float a0 = FLT_MAX, a1 = -FLT_MAX, c0 = FLT_MAX, c1 = -FLT_MAX;
    for (auto& p : pts) {
        float a = glm::dot(p, u), c = glm::dot(p, v);
        a0 = std::min(a0, a); a1 = std::max(a1, a);
        c0 = std::min(c0, c); c1 = std::max(c1, c);
    }
    glm::vec2 cs[4] = {u * a0 + v * c0, u * a1 + v * c0,
                       u * a1 + v * c1, u * a0 + v * c1};
    for (int i = 0; i < 4; ++i) {
        outX[i] = roundf(cs[i].x * 10.0f) / 10.0f;
        outY[i] = roundf(cs[i].y * 10.0f) / 10.0f;
    }
}

/// Axis of the minimum-area box around the points, for the end-on case where
/// the dart has no usable projected axis to align to. Returns false when the
/// points span no area at all.
static bool minAreaBoxAxis(const std::vector<glm::vec2>& pts, glm::vec2& outU) {
    std::vector<glm::vec2> hull = convexHull2D(pts);
    if (hull.size() < 3) return false;
    float bestArea = FLT_MAX;
    glm::vec2 bestU(1.0f, 0.0f);
    for (size_t e = 0; e < hull.size(); ++e) {
        glm::vec2 edge = hull[(e + 1) % hull.size()] - hull[e];
        float len = glm::length(edge);
        if (len < 1e-9f) continue;
        glm::vec2 u = edge / len, v(-u.y, u.x);
        float a0 = FLT_MAX, a1 = -FLT_MAX, c0 = FLT_MAX, c1 = -FLT_MAX;
        for (auto& h : hull) {
            float a = glm::dot(h, u), c = glm::dot(h, v);
            a0 = std::min(a0, a); a1 = std::max(a1, a);
            c0 = std::min(c0, c); c1 = std::max(c1, c);
        }
        float area = (a1 - a0) * (c1 - c0);
        if (area < bestArea) { bestArea = area; bestU = u; }
    }
    outU = bestU;
    return true;
}

// ---------------------------------------------------------------------------
// Compute full frame annotation
// ---------------------------------------------------------------------------

FrameAnnotation computeAnnotation(
    int frameId,
    const FrameState& state,
    const RingRadii& radii,
    double boardZ,
    float hAperture,
    int width, int height,
    const DartGeometry* geom) {

    FrameAnnotation ann;
    ann.frameId = frameId;

    char buf[32];
    snprintf(buf, sizeof(buf), "rgb/%06d.jpg", frameId);
    ann.imagePath = buf;

    ann.focalLength = state.camera.focalLength;
    ann.width = width;
    ann.height = height;
    ann.numDarts = state.numDarts;
    ann.boardRotationDeg = glm::degrees(state.boardRotation);
    ann.boardFaceRotationDeg = glm::degrees(state.boardFaceRotation);
    ann.lightRingMode = state.lightRingMode;
    ann.numLights = (state.light2Intensity > 0.0f) ? 2 : 1;
    ann.ambientLevel = state.ambient;

    // The raster pass's projection without its Vulkan Y-flip, so pixel rows
    // count down from the top as they do in the image. One function builds
    // both, so labels and pixels agree at every aspect ratio.
    const glm::mat4 projMatrix = cameraProjection(
        state.camera.focalLength, hAperture, width, height,
        ann.nearPlane, ann.farPlane);

    // Same lookAt-derived view matrix used for rendering: the inverse of the
    // camera's column-major world matrix (columns = right, up, -forward, eye).
    const glm::mat4 viewMatrix = state.camera.viewMatrix;
    ann.viewMatrix = viewMatrix;

    // Board keypoints follow the NUMBER ring, not the face.
    //
    // They are named by segment ("double_20"), and what makes a position
    // segment 20 is the numeral beside it -- that is what a player reads and
    // what scoring depends on. The numerals are separate meshes drawn at
    // boardRotation only (renderer_lib.cpp), while the face gets
    // boardRotation + boardFaceRotation.
    //
    // boardFaceRotation is always a multiple of 36 degrees (two segments), so
    // including it would not move the keypoints geometrically -- it would only
    // shift which position receives which label, offsetting the labels from
    // the numerals by a random multiple of two segments per frame. Segment
    // identity would then be unlearnable from the numerals, and the only
    // feature tracking the labels would be the logo printed on the face, which
    // transfers to no other board.
    //
    // A future face texture that bakes in its own number ring would invert
    // this: its numerals would rotate with the face, so the keypoints should
    // follow the face and the numeral meshes should be hidden. That needs a
    // per-texture flag rather than a global choice.
    auto keypoints3D = getBoardKeypoints3D(radii, boardZ, state.boardRotation);

    std::vector<glm::vec3> pts;
    pts.reserve(keypoints3D.size());
    for (auto& [name, pos] : keypoints3D) pts.push_back(pos);

    auto projected = projectPoints3D(pts, viewMatrix, projMatrix, width, height);

    ann.boardKeypoints.resize(keypoints3D.size());
    for (size_t i = 0; i < keypoints3D.size(); ++i) {
        ann.boardKeypoints[i] = {
            keypoints3D[i].first,
            roundf(projected[i].x * 10.0f) / 10.0f,
            roundf(projected[i].y * 10.0f) / 10.0f,
        };
    }

    // Un-rotate into number-ring space, not face space, for the same reason the
    // keypoints do: a dart's score is the value of the numeral beside the wedge
    // it landed in, and the numerals sit at boardRotation. The wedge boundaries
    // themselves are unaffected either way -- boardFaceRotation is a multiple of
    // two segments, so the wedge pattern maps onto itself -- so this only
    // decides which value each wedge is assigned.
    float cosRot = cosf(-state.boardRotation);
    float sinRot = sinf(-state.boardRotation);

    for (int i = 0; i < state.numDarts; ++i) {
        const glm::mat4& M = state.dartTransforms[i];

        // Dart origin in world space (translation column of transform)
        glm::vec3 origin(M[3][0], M[3][1], M[3][2]);

        // Dart tip direction: local -Z axis transformed to world space
        // (the dart tip points towards the board, i.e. towards lower Z)
        glm::vec3 tipDir = -glm::vec3(M[2][0], M[2][1], M[2][2]);

        // Line-plane intersection: find where dart axis meets board surface (z = boardZ)
        float t = (static_cast<float>(boardZ) - origin.z) / tipDir.z;
        glm::vec3 boardHit(origin.x + t * tipDir.x, origin.y + t * tipDir.y, static_cast<float>(boardZ));

        // Rear tip of the flight in world space
        glm::vec3 flightTip = boardHit;
        bool flightInFront = false;
        if (geom != nullptr) {
            glm::vec4 tailWorld = M * glm::vec4(geom->tailLocal, 1.0f);
            flightTip = glm::vec3(tailWorld);
            flightInFront = (projMatrix * viewMatrix * tailWorld).w > 0.0f;
        }

        std::vector<glm::vec3> dartPts = {boardHit, flightTip};
        auto dart2D = projectPoints3D(dartPts, viewMatrix, projMatrix, width, height);

        // The oriented box, measured on the PROJECTED geometry.
        //
        // Building it from tip, flight and a perpendicular is wrong under
        // perspective: a 3D direction across the dart's axis does not project
        // to a 2D direction across the projected axis, so the fins' radial
        // extent leaks into the along-axis coordinate. It diverges as the dart
        // turns toward the camera -- the projected length collapses while the
        // flight keeps its 2D size. Such a box leaves 8.6% of dart pixels
        // outside it on average, up to 42%.
        //
        // Projecting the dart's own 3D bounds and taking extents in the
        // projected frame has neither problem, and is a true bound: the hull of
        // the projected corners contains the projection of everything inside.
        float bx[4] = {0, 0, 0, 0}, by[4] = {0, 0, 0, 0};
        bool boxEndOn = false;
        if (geom != nullptr && !geom->verts.empty()) {
            const auto& verts = geom->verts;
            std::vector<glm::vec3> world;
            world.reserve(verts.size());
            // Clipped at the board face. The steel point is buried in the
            // board and therefore occluded, so it contributes no pixels.
            // Fitting to every vertex would leave the box's tip end ~8px
            // (p90 13, max 28) past anything visible and past the landing
            // point that scoring uses.
            //
            // That overhang is not learnable either: penetration is sampled
            // N(7mm, 3mm) per dart and hidden, so a detector asked to place
            // the box's tip end would have to guess a random quantity no
            // pixel reports.
            //
            // boardHit is appended so the box reaches the face exactly. It is
            // where the dart's visible silhouette ends, measured to within
            // 1.4px median of the extreme silhouette pixel.
            for (auto& lp : verts) {
                glm::vec3 w(M * glm::vec4(lp, 1.0f));
                if (w.z >= static_cast<float>(boardZ)) world.push_back(w);
            }
            world.push_back(boardHit);
            auto proj = projectPoints3D(world, viewMatrix, projMatrix,
                                        width, height);
            glm::vec2 tip2(dart2D[0].x, dart2D[0].y);
            glm::vec2 fl2(dart2D[1].x, dart2D[1].y);
            glm::vec2 axis2 = fl2 - tip2;
            float len2 = glm::length(axis2);
            if (len2 > 1e-3f) {
                orientedBoxAlong(proj, axis2 / len2, bx, by);
            } else {
                // End-on: tip and flight project to the same spot, so there
                // is no axis to align to and the box is the minimum-area one.
                // Of its four axis directions, the one nearest whatever is
                // left of the projected axis is taken as tip->flight, so the
                // corner order is still tip-left, flight-left, flight-right,
                // tip-right when there is any direction to honour. With none
                // the choice is arbitrary, which boxEndOn records either way.
                boxEndOn = true;
                glm::vec2 u(1.0f, 0.0f);
                if (minAreaBoxAxis(proj, u)) {
                    const glm::vec2 cands[4] = {u, -u, glm::vec2(-u.y, u.x),
                                                glm::vec2(u.y, -u.x)};
                    glm::vec2 best = u;
                    float bestDot = -FLT_MAX;
                    for (const glm::vec2& c : cands) {
                        const float d = glm::dot(c, axis2);
                        if (d > bestDot) { bestDot = d; best = c; }
                    }
                    u = best;
                }
                orientedBoxAlong(proj, u, bx, by);
            }
        }

        // Un-rotate board hit point to board-local space for score zone
        float localX = boardHit.x * cosRot - boardHit.y * sinRot;
        float localY = boardHit.x * sinRot + boardHit.y * cosRot;

        ann.darts.push_back({
            roundf(dart2D[0].x * 10.0f) / 10.0f,
            roundf(dart2D[0].y * 10.0f) / 10.0f,
            roundf(dart2D[1].x * 10.0f) / 10.0f,
            roundf(dart2D[1].y * 10.0f) / 10.0f,
            flightInFront,
            {bx[0], bx[1], bx[2], bx[3]},
            {by[0], by[1], by[2], by[3]},
            boxEndOn,
            computeScoreZone(localX, localY, radii),
        });
    }

    return ann;
}

// ---------------------------------------------------------------------------
// Dart framing
// ---------------------------------------------------------------------------

bool allDartPointsInFrame(const FrameAnnotation& ann, int width, int height) {
    auto inFrame = [width, height](float x, float y) {
        return x >= 0.0f && x <= (float)width && y >= 0.0f && y <= (float)height;
    };

    for (const auto& d : ann.darts) {
        // The LANDING POINT must be in frame. It is what scoring reads and
        // what the tip head regresses, so a dart whose landing point is
        // outside the crop has no usable target and the frame is redrawn.
        if (!inFrame(d.x, d.y)) return false;
        // Still required: a flight behind the camera means the dart is not
        // being viewed from the front at all, and its box is meaningless.
        if (!d.flightInFront) return false;
        // The flight END is deliberately NOT required to be in frame.
        //
        // The board is allowed to overrun the crop, so a dart near the edge
        // routinely has its flight clipped -- a case real captures produce,
        // not one to reject. Rejecting it would also bias the framing
        // distribution toward centred, well-inside boards.
    }
    return true;
}

FrameState randomizeWithDartsInFrame(
    Randomizer& randomizer,
    int frameId,
    const RingRadii& radii,
    double boardZ,
    float hAperture,
    int width, int height,
    const std::vector<DartGeometry>& dartVariants,
    FrameAnnotation& outAnnotation,
    int maxAttempts) {

    FrameState state;
    for (int attempt = 0; attempt < maxAttempts; ++attempt) {
        state = randomizer.randomize(radii, boardZ, dartVariants);
        // The randomizer chose the design; the annotator has to be told which,
        // because the tail keypoint, the flight half-extent and the fitted
        // oriented box are all properties of that design.
        const DartGeometry* geom =
            dartVariants.empty() ? nullptr
                                 : &dartVariants[std::min(
                                       (size_t)std::max(state.dartVariant, 0),
                                       dartVariants.size() - 1)];
        outAnnotation = computeAnnotation(
            frameId, state, radii, boardZ, hAperture, width, height, geom);
        if (allDartPointsInFrame(outAnnotation, width, height)) break;
    }
    return state;
}

// ---------------------------------------------------------------------------
// JSON serialization
// ---------------------------------------------------------------------------

nlohmann::json annotationToJson(const FrameAnnotation& ann) {
    nlohmann::json j;
    j["frame_id"] = ann.frameId;
    j["image_path"] = ann.imagePath;

    j["board_keypoints"] = nlohmann::json::array();
    for (auto& kp : ann.boardKeypoints) {
        j["board_keypoints"].push_back({
            {"name", kp.name},
            {"x", kp.x},
            {"y", kp.y},
        });
    }

    j["darts"] = nlohmann::json::array();
    for (auto& d : ann.darts) {
        j["darts"].push_back({
            {"x", d.x},
            {"y", d.y},
            {"flight_x", d.flightX},
            {"flight_y", d.flightY},
            {"flight_in_front", d.flightInFront},
            {"box_x", std::vector<float>(d.boxX, d.boxX + 4)},
            {"box_y", std::vector<float>(d.boxY, d.boxY + 4)},
            {"box_end_on", d.boxEndOn},
            {"score_zone", d.scoreZone},
        });
    }

    // Everything needed to rebuild the projection (cameraProjection in
    // board_transforms.h) and so to unproject the depth buffer: without the
    // pose and clip planes it is window-space and uninterpretable outside the
    // renderer.
    std::vector<float> vm;
    vm.reserve(16);
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) vm.push_back(ann.viewMatrix[c][r]);
    j["camera_params"] = {
        {"focal_length", ann.focalLength},
        {"h_aperture_mm", H_APERTURE_MM},
        {"resolution", {ann.width, ann.height}},
        {"view_matrix", vm},              // column-major, glm order
        {"near", ann.nearPlane},
        {"far", ann.farPlane},
    };

    j["metadata"] = {
        {"num_darts", ann.numDarts},
        {"synthetic", true},
        {"board_rotation_deg", ann.boardRotationDeg},
        {"board_face_rotation_deg", ann.boardFaceRotationDeg},
        {"light_ring_mode", ann.lightRingMode},
        {"num_lights", ann.numLights},
        {"ambient_level", ann.ambientLevel},
        {"generator", "vulkan_renderer"},
    };

    return j;
}

} // namespace dart
