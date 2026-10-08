// The board face, evaluated rather than sampled.
//
// Shared by surface_raster.frag and surface_rt.frag, so the painted beds are
// defined once and always agree with the generated wire frame: both are
// derived from the same ring radii. A baked texture can drift from the wires
// when the radii change, and a model trained on that board learns the offset.
//
// Evaluating the face analytically also keeps bed boundaries exact at any
// zoom. Those boundaries are what the network localises, and bilinear
// filtering of a texture smears them across a pixel or two.

#ifndef BOARD_FACE_GLSL
#define BOARD_FACE_GLSL

// The palette arrives per frame in the UBO, LINEAR, as a mat4 whose columns
// are black bed, cream bed, ring-on-black and ring-on-cream. It varies per
// frame so the face is a property the network must read rather than a fixed
// texture it can memorise.
//
// Converted from sRGB to linear once, in the randomizer; the lighting expects
// linear colour and nothing here converts it.

/// Bed colour at board-local `p`, in board units (1 BU = 100mm), bull at the
/// origin. Radii mirror constants.h; they are the same numbers the wires, the
/// annotations and the app's scoring all use.
vec3 boardFaceAlbedo(vec2 p, mat4 palette) {
    vec3 kBedBlack  = palette[0].rgb;
    vec3 kBedCream  = palette[1].rgb;
    vec3 kRingRed   = palette[2].rgb;
    vec3 kRingGreen = palette[3].rgb;
    float r = length(p);

    // Angle from +Y increasing toward +X, matching the renderer's keypoint
    // convention (x = R sin t, y = R cos t), which is what puts 20 at the top.
    float theta = mod(degrees(atan(p.x, p.y)), 360.0);
    int seg = int(mod(floor(mod(theta + 9.0, 360.0) / 18.0), 20.0));

    // Even positions in SEGMENT_ORDER are the black beds; the double and
    // treble of a black bed are red, and of a cream bed green.
    bool blackBed = (seg % 2) == 0;
    vec3 bed  = blackBed ? kBedBlack : kBedCream;
    vec3 ring = blackBed ? kRingRed : kRingGreen;

    // Written outward so each test overwrites the last.
    //
    // The annulus outside the double is the same black sisal as a black bed,
    // because on a real board it is literally the same material.
    vec3 c = kBedBlack;
    if (r > 1.07 && r <= 1.62) c = bed;   // outer single
    if (r > 0.99 && r <= 1.07) c = ring;  // treble
    if (r > 0.159 && r <= 0.99) c = bed;  // inner single
    if (r > 1.62 && r <= 1.70) c = ring;  // double
    if (r <= 0.159) c = kRingGreen;       // outer bull
    if (r <= 0.0635) c = kRingRed;        // inner bull
    return c;
}

#endif
