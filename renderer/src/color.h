#pragma once

#include <glm/glm.hpp>

#include <cmath>

namespace dart {

// Every colour handed to a shader is LINEAR. The colour attachments are sRGB
// formats, so the hardware applies the encode on write; a colour authored as
// a display value (a swatch, a byte triple, a draw on the unit cube meant as
// "what it looks like") is decoded here, once, on the CPU. Shader-side
// conversion is easy to apply twice or not at all.

/// Exact piecewise sRGB decode, the same curve an SRGB format applies on
/// sampling.
inline float srgbToLinear(float v) {
    return v <= 0.04045f ? v / 12.92f
                         : std::pow((v + 0.055f) / 1.055f, 2.4f);
}

inline glm::vec3 srgbToLinear(const glm::vec3& c) {
    return glm::vec3(srgbToLinear(c.r), srgbToLinear(c.g), srgbToLinear(c.b));
}

} // namespace dart
