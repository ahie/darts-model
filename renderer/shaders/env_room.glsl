// The room the board is standing in, as seen by reflections and ambient.
//
// The environment probe is the background photograph, sampled as an equirect
// map. That has two problems a dartboard shows off badly.
//
// It is 8-bit sRGB, so nothing in it exceeds 1.0. A real ceiling fixture sits
// one to two orders of magnitude above the wall it lights, and that RATIO is
// what gives a polished barrel or a steel wire a highlight with shape. Clamp
// the brightest thing in the room to the same value as the wall behind it and
// metal reflects a flat wash -- which reads as plastic no matter how correct
// the BRDF is, and the BRDF here is already GGX with height-correlated Smith.
//
// It is also a photograph of ONE direction, smeared over the whole sphere, so
// the floor is as bright as the ceiling. Indoors that is never true, and the
// vertical gradient is most of what tells you a surface is indoors at all.
//
// Neither is fixed by adding the key light to the probe: the key light is
// already an analytic area source with its own GGX response, so putting it
// here as well would double-count it. What is missing is everything ELSE --
// the fixtures, windows and bright walls that no analytic light models.

#ifndef ENV_ROOM_GLSL
#define ENV_ROOM_GLSL

// Own constant: both including shaders define PI, but this header is included
// before their declarations and must not depend on them.
const float PI_ENV = 3.14159265359;

/// Angular radius of the fixture, degrees.
///
/// A 1.2x0.3m panel at 2.5m subtends about 13 by 3 degrees; 7-15 is a generous
/// read of that. A much wider cone behaves as an overcast sky: a polished
/// barrel reflecting a large part of the sphere at 2-22x radiance comes out
/// white no matter what its F0 is.
const float kFixtureInnerDeg = 7.0;
const float kFixtureOuterDeg = 15.0;

/// Solid angle of the outer cone, sr: 2*pi*(1 - cos(15 deg)).
const float kFixtureSolidAngle = 0.2141;

/// The backdrop, with the vertical structure a single photo cannot have.
vec3 roomBackdrop(vec3 photo, float up) {
    // Floor darker, ceiling brighter. Mild, because the photo is a real
    // backdrop and its own content should still dominate the mid-band.
    return photo * mix(0.35, 1.25, up * 0.5 + 0.5);
}

/// Environment radiance for a REFLECTION ray.
///
/// `widenDeg` blurs the fixture for rough surfaces. Physically right -- a
/// brushed barrel smears a light source where a polished one mirrors it -- and
/// it also keeps a small bright source from aliasing into speckle.
vec3 roomEnvSpecular(vec3 photo, vec3 dir, vec4 room, float widenDeg) {
    float up = clamp(dir.y, -1.0, 1.0);
    float ang = degrees(acos(up));
    float outer = kFixtureOuterDeg + widenDeg;
    float lobe = 1.0 - smoothstep(kFixtureInnerDeg + widenDeg, outer, ang);

    // Spreading the source has to dim it, or widening ADDS light. Blurred at
    // constant peak radiance, rough barrels would come out brighter than
    // polished ones: at roughness 1 the widened cone reaches 75 degrees,
    // covering 37% of the sphere.
    float omega = 2.0 * PI_ENV * (1.0 - cos(radians(outer)));
    lobe *= kFixtureSolidAngle / max(omega, 1e-4);

    return roomBackdrop(photo, up) + room.rgb * (room.w * lobe);
}

/// Environment radiance for the IRRADIANCE integral.
///
/// The fixture is spread over the upper hemisphere carrying the same total
/// power, rather than left as the small bright disc the reflection path sees.
/// Irradiance only needs the integral, and sixteen cosine-weighted samples
/// against a source covering 1.7% of the sphere would miss it entirely on most
/// pixels and hit it hard on a few -- which is fireflies, not lighting.
///
/// Energy match: a cosine lobe A*max(up,0) integrates to 2*pi*A/3 against the
/// cosine-weighted measure, and the disc delivers radiance * solidAngle, so
/// A = 3 * solidAngle / (2*pi) = 0.102 per unit radiance.
vec3 roomEnvDiffuse(vec3 photo, vec3 dir, vec4 room) {
    float up = clamp(dir.y, -1.0, 1.0);
    return roomBackdrop(photo, up)
         + room.rgb * (room.w * 3.0 * kFixtureSolidAngle / (2.0 * PI_ENV)
                       * max(up, 0.0));
}

#endif
