#include "board_mesh.h"

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dart {

namespace {

constexpr int kFloats = 12;

/// Append one vertex. Tangent w is the bitangent sign, always +1 here: both
/// caps and the rim use a right-handed (tangent, bitangent, normal) frame.
void push(std::vector<float>& v,
          float px, float py, float pz,
          float nx, float ny, float nz,
          float u, float w,
          float tx, float ty, float tz) {
    v.insert(v.end(), {px, py, pz, nx, ny, nz, u, w, tx, ty, tz, 1.0f});
}

} // namespace

void buildBoardGeometry(const BoardMeshParams& params,
                        std::vector<float>& vertices,
                        std::vector<uint32_t>& indices) {
    const int seg = params.segments < 3 ? 3 : params.segments;
    const int rings = params.faceRings < 1 ? 1 : params.faceRings;
    const float R = params.radiusBU;
    const float zf = params.faceZ;
    const float zb = params.faceZ - params.depthBU;

    const uint32_t base = (uint32_t)(vertices.size() / kFloats);

    // ---- Front face -------------------------------------------------------
    //
    // Concentric rings, centre vertex first. UVs are the board disc mapped into
    // the unit square. Nothing samples them, since the face is analytic, but
    // they cost two floats and keep the vertex format uniform with the other
    // meshes.
    push(vertices, 0.0f, 0.0f, zf, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f);
    for (int r = 1; r <= rings; ++r) {
        const float rad = R * (float)r / (float)rings;
        for (int s = 0; s < seg; ++s) {
            const float a = 2.0f * (float)M_PI * (float)s / (float)seg;
            const float x = rad * std::cos(a), y = rad * std::sin(a);
            push(vertices, x, y, zf, 0.0f, 0.0f, 1.0f,
                 0.5f + x / (2.0f * R), 0.5f + y / (2.0f * R),
                 1.0f, 0.0f, 0.0f);
        }
    }
    // Innermost ring fans off the centre vertex.
    for (int s = 0; s < seg; ++s) {
        const uint32_t a = base + 1 + (uint32_t)s;
        const uint32_t b = base + 1 + (uint32_t)((s + 1) % seg);
        indices.insert(indices.end(), {base, a, b});
    }
    for (int r = 1; r < rings; ++r) {
        const uint32_t inner = base + 1 + (uint32_t)((r - 1) * seg);
        const uint32_t outer = base + 1 + (uint32_t)(r * seg);
        for (int s = 0; s < seg; ++s) {
            const uint32_t s1 = (uint32_t)((s + 1) % seg);
            const uint32_t i0 = inner + (uint32_t)s, i1 = inner + s1;
            const uint32_t o0 = outer + (uint32_t)s, o1 = outer + s1;
            indices.insert(indices.end(), {i0, o0, o1, i0, o1, i1});
        }
    }

    // ---- Rim --------------------------------------------------------------
    //
    // The part that actually decides whether the silhouette reads as round, and
    // the only part of the board a camera sees edge-on at high obliquity --
    // about a quarter of the frames.
    const uint32_t rimBase = (uint32_t)(vertices.size() / kFloats);
    for (int s = 0; s < seg; ++s) {
        const float a = 2.0f * (float)M_PI * (float)s / (float)seg;
        const float c = std::cos(a), sn = std::sin(a);
        const float u = (float)s / (float)seg;
        // Tangent runs along the rim; normal points out of the cylinder.
        push(vertices, R * c, R * sn, zf, c, sn, 0.0f, u, 0.0f, -sn, c, 0.0f);
        push(vertices, R * c, R * sn, zb, c, sn, 0.0f, u, 1.0f, -sn, c, 0.0f);
    }
    for (int s = 0; s < seg; ++s) {
        const uint32_t s1 = (uint32_t)((s + 1) % seg);
        const uint32_t f0 = rimBase + (uint32_t)(s * 2), b0 = f0 + 1;
        const uint32_t f1 = rimBase + s1 * 2, b1 = f1 + 1;
        indices.insert(indices.end(), {f0, b0, b1, f0, b1, f1});
    }

    // ---- Back -------------------------------------------------------------
    //
    // A board hangs on a wall, so this is never seen. It is here so the mesh is
    // closed: an open shell gives ray queries a way into the interior, and an
    // ambient or shadow ray that enters through the rim and exits the missing
    // back face reports occlusion that is not there.
    const uint32_t backBase = (uint32_t)(vertices.size() / kFloats);
    push(vertices, 0.0f, 0.0f, zb, 0.0f, 0.0f, -1.0f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f);
    for (int s = 0; s < seg; ++s) {
        const float a = 2.0f * (float)M_PI * (float)s / (float)seg;
        const float x = R * std::cos(a), y = R * std::sin(a);
        push(vertices, x, y, zb, 0.0f, 0.0f, -1.0f,
             0.5f + x / (2.0f * R), 0.5f - y / (2.0f * R), 1.0f, 0.0f, 0.0f);
    }
    for (int s = 0; s < seg; ++s) {
        const uint32_t a = backBase + 1 + (uint32_t)s;
        const uint32_t b = backBase + 1 + (uint32_t)((s + 1) % seg);
        indices.insert(indices.end(), {backBase, b, a});
    }
}

} // namespace dart
