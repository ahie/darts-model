#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

namespace dart {

// Model transforms shared by the raster pass and the ray-tracing acceleration
// structure.
//
// These two paths have to agree exactly. When they drift, the raster image and
// the geometry that casts its shadows disagree, which shows up as shadows in
// the wrong place rather than as anything that looks like a bug in either path
// alone -- e.g. a numeral flipped only when drawing casts its shadow in the
// unflipped orientation.

/// Model matrix for one numeral of the number ring.
///
/// `pos` is the numeral's board-local position (the translation column of its
/// base transform), `localXform` that transform with the translation stripped,
/// and `scaleM` the ring's scale.
inline glm::mat4 numeralModelMatrix(const glm::mat4& boardRot,
                                    const glm::vec3& pos,
                                    const glm::mat4& scaleM,
                                    const glm::mat4& localXform,
                                    float ringOffset = 0.0f) {
    // Number rings are stamped so the whole ring reads upright with the board
    // head-on and 20 at the top, not so every numeral points radially outward.
    // The upper half -- 11, 14, 9, 12, 5, 20, 1, 18, 4, 13, 6, spanning 270
    // degrees through 0 to 90 -- has its bases toward the centre; the lower
    // nine (10, 15, 2, 17, 3, 19, 7, 16, 8) are turned 180 degrees so they are
    // not upside down, as on every board on the market.
    //
    // Decided from the numeral's own angle rather than its name so a font
    // variant that names its meshes differently still works. The flip is
    // applied in board-local space, before boardRot, because it is a property
    // of how the ring was manufactured -- a board mounted rotated really does
    // show some numerals upside down.
    float numAngle = glm::degrees(std::atan2(pos.x, pos.y));
    if (numAngle < 0.0f) numAngle += 360.0f;
    glm::mat4 numFlip(1.0f);
    if (numAngle > 90.5f && numAngle < 269.5f) {
        numFlip = glm::rotate(glm::mat4(1.0f), glm::pi<float>(),
                              glm::vec3(0, 0, 1));
    }

    // The ring offset turns the whole hoop about the board axis, so it is
    // applied outside the numeral's own placement -- the numerals keep their
    // exact spacing relative to each other and the ring as a unit sits crooked,
    // which is how a hand-seated ring actually fails.
    const glm::mat4 ring =
        glm::rotate(glm::mat4(1.0f), ringOffset, glm::vec3(0, 0, 1));
    return boardRot * ring * glm::translate(glm::mat4(1.0f), pos)
         * scaleM * numFlip * localXform;
}

// Camera projection shared by the raster pass and the annotator.
//
// The same agreement applies: labels are projected on the CPU and pixels on
// the GPU, and any difference between the two matrices is a label offset that
// grows toward the frame edge.

/// Vertical field of view, radians, for a focal length quoted against the
/// HORIZONTAL aperture.
///
/// The aperture spans the frame width, so it fixes the horizontal field;
/// glm::perspective takes the vertical one. For a square frame the two
/// coincide, and for any other aspect passing the horizontal field as the
/// vertical one stretches the image off the labels.
inline float verticalFov(float focalLength, float hAperture, float aspect) {
    const float fovX = 2.0f * std::atan(hAperture / (2.0f * focalLength));
    return 2.0f * std::atan(std::tan(0.5f * fovX) / aspect);
}

/// OpenGL-convention projection (no Y-flip, depth in [-1, 1]) for a frame of
/// width x height. The raster pass negates [1][1] for Vulkan's Y-down clip
/// space; nothing else about the two differs.
inline glm::mat4 cameraProjection(float focalLength, float hAperture,
                                  int width, int height,
                                  float nearP, float farP) {
    const float aspect = (float)width / (float)height;
    return glm::perspective(verticalFov(focalLength, hAperture, aspect),
                            aspect, nearP, farP);
}

} // namespace dart
