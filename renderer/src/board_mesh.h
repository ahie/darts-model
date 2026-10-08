#pragma once

#include <cstdint>
#include <vector>

namespace dart {

// The board itself: a sisal cylinder, generated rather than loaded.
//
// Generated so the tessellation is a parameter: a coarse polygon rim reads as
// visibly faceted against a real photograph, where a board is a clean circle.
// The face colour is evaluated analytically from board position (see
// boardFaceAlbedo in board_face.glsl), so the mesh carries no texture, and
// the painted beds cannot disagree with the wire frame.
struct BoardMeshParams {
    /// Segments around the rim. At 1536 the chord is ~2px with the board
    /// filling a 1024px frame, which is below what the sisal grain resolves.
    int segments = 1536;

    /// Radial subdivisions of the face.
    ///
    /// The face is flat and shaded per-pixel from world position, so this
    /// changes nothing visually; it exists to keep triangles from degenerating
    /// into slivers at the centre of a single fan, which is harmless for
    /// rendering but makes the mesh unusable for anything that interpolates
    /// across it.
    int faceRings = 3;

    /// Board radius, and how deep the cylinder runs behind its face.
    ///
    /// 38mm is the bristle-board spec: the cylinder runs z -0.19 to +0.19,
    /// symmetric about zero with its face at BOARD_SURFACE_Z. Thickness is
    /// invisible head-on but obvious the moment the camera is off-axis, which
    /// is the common case since the orbit reaches 70 degrees of obliquity.
    float radiusBU = 2.255f;
    float depthBU = 0.38f;

    /// Z of the front face.
    float faceZ = 0.19f;
};

/// Build the board into interleaved vertex / index buffers, appending to them.
///
/// Layout matches the renderer's pipeline and wire_geometry's:
/// position(3) normal(3) uv(2) tangent(4) = 12 floats.
void buildBoardGeometry(const BoardMeshParams& params,
                        std::vector<float>& vertices,
                        std::vector<uint32_t>& indices);

} // namespace dart
