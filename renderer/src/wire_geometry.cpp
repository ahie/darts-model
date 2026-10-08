#include "wire_geometry.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include <cmath>

namespace dart {

const char* const kRingWireRadii[6] = {
    "inner_bull_r", "outer_bull_r",
    "triple_inner_r", "triple_outer_r",
    "double_inner_r", "double_outer_r",
};

namespace {

void pushVertex(std::vector<float>& v, const glm::vec3& pos, const glm::vec3& nrm,
                const glm::vec3& tangent) {
    v.push_back(pos.x); v.push_back(pos.y); v.push_back(pos.z);
    v.push_back(nrm.x); v.push_back(nrm.y); v.push_back(nrm.z);
    // The wires carry no texture; a dummy white sampler is bound instead.
    v.push_back(0.0f); v.push_back(0.0f);
    v.push_back(tangent.x); v.push_back(tangent.y); v.push_back(tangent.z);
    v.push_back(1.0f);
}

/// Cross-section offset and its normal at parameter `a` around the wire.
/// `e1`/`e2` span the plane perpendicular to the wire's run.
void crossSection(const WireParams& p, float a, const glm::vec3& e1, const glm::vec3& e2,
                  glm::vec3& offset, glm::vec3& normal) {
    const float r = p.thickness * 0.5f;
    if (!p.blade) {
        const float c = std::cos(a), s = std::sin(a);
        offset = r * (c * e1 + s * e2);
        normal = c * e1 + s * e2;
        return;
    }
    // Blade wire: a thin rectangle standing on edge. Walk the perimeter and
    // emit the face normal of whichever side the parameter falls on.
    const float half = r;
    const float tall = r * p.bladeHeightRatio;
    const float t = a / glm::two_pi<float>() * 4.0f;  // 0..4 around the rectangle
    const int side = (int)t % 4;
    const float f = t - std::floor(t);
    switch (side) {
        case 0: offset =  half * e1 + (-tall + 2 * tall * f) * e2; normal =  e1; break;
        case 1: offset = ( half - 2 * half * f) * e1 +  tall * e2; normal =  e2; break;
        case 2: offset = -half * e1 + ( tall - 2 * tall * f) * e2; normal = -e1; break;
        default: offset = (-half + 2 * half * f) * e1 - tall * e2; normal = -e2; break;
    }
}

/// Sweep a cross-section along a path sampled by `pointAt`, producing a tube.
/// `closed` joins the last ring back to the first.
template <typename PathFn>
void sweep(const WireParams& p, int steps, bool closed, PathFn pointAt,
           std::vector<float>& verts, std::vector<uint32_t>& idx) {
    const uint32_t base = (uint32_t)(verts.size() / kFloatsPerVertex);
    const int rings = closed ? steps : steps + 1;

    for (int i = 0; i < rings; ++i) {
        glm::vec3 centre, dir;
        pointAt(i, centre, dir);
        // Build a stable frame perpendicular to the wire direction.
        glm::vec3 up(0.0f, 0.0f, 1.0f);
        if (std::abs(glm::dot(dir, up)) > 0.99f) up = glm::vec3(1.0f, 0.0f, 0.0f);
        const glm::vec3 e1 = glm::normalize(glm::cross(dir, up));
        const glm::vec3 e2 = glm::normalize(glm::cross(e1, dir));
        for (int k = 0; k < p.tubeSegments; ++k) {
            const float a = glm::two_pi<float>() * (float)k / (float)p.tubeSegments;
            glm::vec3 off, nrm;
            crossSection(p, a, e1, e2, off, nrm);
            pushVertex(verts, centre + off, glm::normalize(nrm), dir);
        }
    }

    const int T = p.tubeSegments;
    const int quads = closed ? steps : steps;
    for (int i = 0; i < quads; ++i) {
        const int i0 = i;
        const int i1 = closed ? (i + 1) % steps : i + 1;
        for (int k = 0; k < T; ++k) {
            const int k1 = (k + 1) % T;
            const uint32_t a = base + i0 * T + k;
            const uint32_t b = base + i1 * T + k;
            const uint32_t c = base + i1 * T + k1;
            const uint32_t d = base + i0 * T + k1;
            idx.push_back(a); idx.push_back(b); idx.push_back(c);
            idx.push_back(a); idx.push_back(c); idx.push_back(d);
        }
    }
}

double lookup(const std::unordered_map<std::string, double>& m, const char* k) {
    auto it = m.find(k);
    return it == m.end() ? 0.0 : it->second;
}

} // namespace

void buildSpiderGeometry(const std::unordered_map<std::string, double>& radiiBU,
                         double boardZ,
                         const WireParams& p,
                         std::vector<float>& vertices,
                         std::vector<uint32_t>& indices) {
    const float zc = (float)boardZ + p.standoff;

    // --- Ring wires -------------------------------------------------------
    for (int ri = 0; ri < 6; ++ri) {
        const float Rc = (float)lookup(radiiBU, kRingWireRadii[ri]);
        if (Rc <= 0.0f) continue;
        sweep(p, p.ringSegments, /*closed=*/true,
              [&](int i, glm::vec3& c, glm::vec3& d) {
                  const float t = glm::two_pi<float>() * (float)i / (float)p.ringSegments;
                  c = glm::vec3(Rc * std::sin(t), Rc * std::cos(t), zc);
                  // Tangent runs along the ring.
                  d = glm::normalize(glm::vec3(std::cos(t), -std::sin(t), 0.0f));
              },
              vertices, indices);
    }

    // --- Radial wires -----------------------------------------------------
    // One per segment boundary, spanning the outer bull to the outer double.
    const float r0 = (float)lookup(radiiBU, "outer_bull_r");
    const float r1 = (float)lookup(radiiBU, "double_outer_r");
    if (r1 > r0) {
        const float segSpan = glm::two_pi<float>() / 20.0f;
        for (int s = 0; s < 20; ++s) {
            // Segment i is centred on i*18deg (20 at the top, +Y), so the
            // boundaries fall half a segment round from each centre.
            const float th = segSpan * ((float)s + 0.5f);
            const glm::vec3 axis(std::sin(th), std::cos(th), 0.0f);
            sweep(p, 2, /*closed=*/false,
                  [&](int i, glm::vec3& c, glm::vec3& d) {
                      const float f = (float)i / 2.0f;
                      c = glm::vec3(axis * (r0 + (r1 - r0) * f));
                      c.z = zc;
                      d = axis;
                  },
                  vertices, indices);
        }
    }
}


WireParams spiderVariant(int v) {
    WireParams w;
    switch (v) {
        case 1:  // fine round staple
            w.thickness = 0.0038f; w.standoff = 0.0050f; break;
        case 2:  // heavier round staple, older boards
            w.thickness = 0.0064f; w.standoff = 0.0076f; break;
        case 3:  // thin blade standing tall
            w.thickness = 0.0022f; w.standoff = 0.0046f;
            w.blade = true; w.bladeHeightRatio = 3.2f; break;
        case 4:  // medium blade
            w.thickness = 0.0031f; w.standoff = 0.0056f;
            w.blade = true; w.bladeHeightRatio = 2.5f; break;
        case 0:
        default: // the default round staple
            break;
    }
    return w;
}

} // namespace dart
