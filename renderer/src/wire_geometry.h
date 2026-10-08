#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dart {

// Procedural dartboard "spider": the wire frame that separates the beds.
//
// Modelled as geometry rather than painted into the texture because the wires
// are the highest-contrast features on a real board and sit exactly on the ring
// boundaries the network has to localise. Painted wires would also flatten
// under the board's own lighting, losing the specular edge that makes them
// readable in a photograph.
//
// Placement convention
// --------------------
// The WDF/BDO spec quotes the ring radii to wire EDGES: 107mm and 170mm are
// from the centre of the bull to the outside edge of the treble's and the
// double's OUTER wires, with each bed 8mm wide inside them (constants.h).
//
// Wires are centred on the RING_RADII_BU boundaries instead, which is also
// what every scoring convention in this codebase uses -- the annotation
// radii, the analytic board face, and the Python package's RING_RADII. The
// spec's edge measurements are then met to within half a wire thickness
// (0.1-0.3mm across the spider variants), the beds stay centred on their
// nominal radii, and the wire straddles the colour transition, which is what
// a real board looks like.
struct WireParams {
    /// Wire diameter, board units (1 BU = 100mm).
    float thickness = 0.005f;

    /// Height of the wire's centre above the board face.
    ///
    /// Set above half the thickness, so the wire stands clear of the face
    /// rather than resting tangent to it. That is what a real board does: the
    /// spider is stapled on top of the sisal, not bedded into it, so there is a
    /// small gap underneath and the wire catches light along its whole
    /// circumference. At 0.6mm with a 0.5mm wire the crown sits 0.85mm proud.
    ///
    /// Raising this is the way to get more protrusion from a thin wire --
    /// thickening it instead reads far too heavy against the 8mm beds. For a
    /// thin-but-tall profile, `blade` is the more accurate model.
    float standoff = 0.006f;

    /// Rectangular cross-section (blade) rather than round (staple).
    bool blade = false;

    /// For blade wire, how far it stands proud relative to its thickness.
    float bladeHeightRatio = 2.5f;

    /// Tessellation around a full ring, and around the wire cross-section.
    int ringSegments = 512;
    int tubeSegments = 8;
};

/// Parameters for spider variant `v`, 0 <= v < kSpiderVariants.
///
/// Real boards differ in what the spider is made of, and it changes the look
/// far more than the colour does. Traditional boards use round staple wire; the
/// modern ones use a thin blade standing proud of the face, which casts a much
/// narrower shadow and catches light on its edge rather than its crown. The
/// round gauges stay near the 0.5mm that reads correctly against 8mm beds --
/// thicker wire looks heavy and wrong long before it looks sturdy.
WireParams spiderVariant(int v);

/// Which radii carry a ring wire, in the order they appear outward.
extern const char* const kRingWireRadii[6];

/// Interleaved vertex layout matching the renderer's pipeline:
/// position(3) normal(3) uv(2) tangent(4) = 12 floats.
constexpr int kFloatsPerVertex = 12;

/// Build the spider into interleaved vertex / index buffers.
///
/// `radiiBU` holds RING_RADII_BU from constants.h, keyed by field name. `boardZ` is
/// the board's front face. Appends to the supplied vectors.
void buildSpiderGeometry(const std::unordered_map<std::string, double>& radiiBU,
                         double boardZ,
                         const WireParams& params,
                         std::vector<float>& vertices,
                         std::vector<uint32_t>& indices);

} // namespace dart
