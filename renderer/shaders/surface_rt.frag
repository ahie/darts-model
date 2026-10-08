#version 460
#extension GL_EXT_ray_query : require
#extension GL_GOOGLE_include_directive : require

// Set layout: 0 UBO, 1 albedo, 2 normal map, 3 TLAS + materials,
// 4 environment, 5 decal atlas.
#define DECAL_SET 5

#include "board_face.glsl"
#include "env_room.glsl"
#include "surface_common.glsl"

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;
layout(location = 3) flat in uint fragDrawMode;
layout(location = 4) flat in vec3 fragColor;
layout(location = 5) in vec3 fragTangent;
layout(location = 6) in float fragBitangentSign;

layout(location = 0) out vec4 outColor;

// Texture sampler
layout(set = 1, binding = 0) uniform sampler2D texSampler;

// Normal map sampler
layout(set = 2, binding = 0) uniform sampler2D normalSampler;

// TLAS for ray queries
layout(set = 3, binding = 0) uniform accelerationStructureEXT topLevelAS;

// One entry per TLAS instance, indexed by instanceCustomIndex. A ray query
// returns an instance index and nothing else, so this is the only way the
// shader can find out what a ray actually hit.
struct RayMaterial {
    vec4 albedo;    // rgb = base colour, a = kind (0 flat, 1 board face, 2 flight)
    vec4 transmit;  // rgb = what survives passing through
};
// std430 explicitly: the default for a buffer block is `shared`, whose layout
// is implementation-defined, and this struct is written by matching C++.
layout(std430, set = 3, binding = 1) readonly buffer Materials {
    RayMaterial rayMaterials[];
};

// The background image doubles as a reflection probe: reflection rays that
// escape the scene need something to return, and a board plus three darts in
// empty space means most of them escape.
layout(set = 4, binding = 0) uniform sampler2D envSampler;

// --------------------------------------------------------------------------
// Sample directions for AO, shadow and reflection rays. The hash is the
// sisal one: any well-mixed 2D hash serves, and one copy is enough.
// --------------------------------------------------------------------------
vec3 cosineHemisphere(vec3 N, vec2 seed) {
    float u1 = sisalHash(seed);
    float u2 = sisalHash(seed + vec2(17.0, 31.0));

    float r = sqrt(u1);
    float theta = 6.28318530718 * u2;

    // Build tangent frame
    vec3 up = abs(N.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
    vec3 T = normalize(cross(up, N));
    vec3 B = cross(N, T);

    return normalize(T * (r * cos(theta)) + B * (r * sin(theta)) + N * sqrt(1.0 - u1));
}

// --------------------------------------------------------------------------
// Ray query helpers
// --------------------------------------------------------------------------

// Colour a ray query hit returns to whatever is looking at it.
//
// The board face gets special handling because it is a single instance
// covering every bed: its world hit position is pushed back into the face's
// own frame and evaluated there, so the reflection picks up the actual bed it
// struck. Everything else carries a flat colour, which is what its raster draw
// uses anyway.
vec3 rayHitAlbedo(int matIdx, vec3 hitPos) {
    RayMaterial m = rayMaterials[matIdx];
    if (m.albedo.a > 0.5 && m.albedo.a < 1.5) {
        vec3 local = (scene.boardFaceInvModel * vec4(hitPos, 1.0)).xyz;
        return boardFaceAlbedo(local.xy, scene.boardPalette);
    }
    return m.albedo.rgb;
}

// Shadow transmission through translucent dart flights.
//
// Flights are the one thing on the board that light passes through, and a
// tinted one tints what passes -- so the shadow takes the flight's colour
// rather than merely being lighter. Their BLAS geometry is built non-opaque
// so candidate hits are reported here instead of terminating the ray; anything
// else committed is a solid blocker. customIndex is the hit instance's slot in
// rayMaterials, which says whether it is a flight and what it transmits.
vec3 shadowTransmission(vec3 origin, vec3 dir, float tMax) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsNoneEXT,
                          0xFF, origin, 0.002, dir, tMax);

    vec3 transmit = vec3(1.0);
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false)
                == gl_RayQueryCandidateIntersectionTriangleEXT) {
            int idx = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false);
            if (rayMaterials[idx].albedo.a > 1.5) {
                // Translucent flight: attenuate and keep going. Not committed,
                // so the ray continues past it to whatever lies beyond.
                transmit *= rayMaterials[idx].transmit.rgb;
                if (dot(transmit, vec3(1.0)) < 0.01) {
                    rayQueryConfirmIntersectionEXT(rq);  // effectively opaque now
                }
            } else {
                rayQueryConfirmIntersectionEXT(rq);
            }
        }
    }

    if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT)
        return vec3(0.0);
    return transmit;
}

// `nSamples` is a parameter so two fixtures SPLIT one ray budget rather than
// doubling it: 40 for the key and 24 for the second. Two shadow directions
// with slightly noisier penumbrae are a better trade than one clean shadow,
// because a lone shadow direction is one of the things that reads as rendered.
vec3 traceShadow(vec3 origin, vec3 lightPos, int nSamples) {

    // Soft shadow: jitter light position to simulate area light
    int NUM_SHADOW_SAMPLES = nSamples;
    const float LIGHT_RADIUS = 1.5;  // BU — large area light for soft penumbra
    vec3 litRGB = vec3(0.0);
    // Emission-weighted, so shadow rays are not cast from a dark part of the
    // ring. Every weight is 1 on the point path, where this is a plain mean.
    float wsum = 0.0;

    for (int i = 0; i < NUM_SHADOW_SAMPLES; i++) {
        vec2 seed = gl_FragCoord.xy * 0.53 + vec2(float(i) * 5.17, float(i) * 13.29);
        vec3 jitteredLight;
        float w = 1.0;
        if (scene.lightRing.x > 0.5) {
            // Ring fixture around the board: sample a point on the ring itself
            // rather than a blob around a single position. Occluders are lit
            // from every side at once, so the casts overlap and cancel instead
            // of producing one directional shadow -- which is exactly why a
            // ring-lit board looks nearly shadowless.
            float a = 6.28318530718 * (float(i) / float(NUM_SHADOW_SAMPLES)
                                       + sisalHash(seed) * 0.02);
            float r = scene.lightRing.y;
            w = ringEmission(a);
            jitteredLight = vec3(cos(a) * r, sin(a) * r, scene.lightRing.z);
        } else {
            // A disc facing the shaded point, not a box around the light: a
            // box's penumbra width depends on which way the light sits relative
            // to the surface, so softness would vary with geometry rather than
            // with the fixture. sqrt() keeps the samples area-uniform instead
            // of piling them at the centre.
            vec3 toLight = normalize(lightPos - origin);
            vec3 ref = (abs(toLight.z) < 0.999) ? vec3(0.0, 0.0, 1.0)
                                                : vec3(1.0, 0.0, 0.0);
            vec3 dt = normalize(cross(ref, toLight));
            vec3 db = cross(toLight, dt);
            float rr = LIGHT_RADIUS * 0.5 * sqrt(sisalHash(seed));
            float aa = 6.28318530718 * sisalHash(seed + vec2(3.0, 7.0));
            jitteredLight = lightPos + (dt * cos(aa) + db * sin(aa)) * rr;
        }

        wsum += w;
        if (w <= 0.0) continue;

        vec3 dir = jitteredLight - origin;
        float tMax = length(dir);
        dir /= tMax;

        litRGB += w * shadowTransmission(origin, dir, tMax);
    }

    return litRGB / max(wsum, 1e-4);
}

float traceAO(vec3 origin, vec3 N, float radius) {
    float occlusion = 0.0;
    const int NUM_SAMPLES = 16;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        vec2 seed = gl_FragCoord.xy * 0.37 + vec2(float(i) * 7.13, float(i) * 11.71);
        vec3 dir = cosineHemisphere(N, seed);

        // Transmission walk rather than a binary opaque test, matching
        // traceAmbient: a translucent flight should not cast a solid contact
        // shadow when it already passes tinted light everywhere else.
        occlusion += 1.0 - dot(shadowTransmission(origin, dir, radius),
                               vec3(0.33333));
    }

    return 1.0 - (occlusion / float(NUM_SAMPLES));
}

// Ambient light gathered from the same rays that measure occlusion.
//
// Occlusion needs 16 cosine-weighted rays per fragment anyway; keeping what
// the misses see turns the hit/miss count into a real irradiance estimate at
// no additional ray cost. The directions are cosine-distributed, so the mean
// radiance over unoccluded samples IS the diffuse irradiance factor, and
// occluded samples contribute zero, which folds the occlusion term in for free.
//
// A flat scalar ambient would be the same neutral grey on every surface
// regardless of orientation while the key light ranges over 3000-7000K: a fill
// nothing in the scene casts, which flattens shading and pulls every surface
// toward the same tint. Gathered this way, ambient has a direction, a colour,
// and a magnitude that tracks the room the board is standing in.
void traceAmbient(vec3 origin, vec3 N, float radius,
                  out float ao, out vec3 irradiance) {
    const int NUM_SAMPLES = 16;
    float open = 0.0;
    vec3 env = vec3(0.0);

    for (int i = 0; i < NUM_SAMPLES; i++) {
        vec2 seed = gl_FragCoord.xy * 0.37 + vec2(float(i) * 7.13, float(i) * 11.71);
        vec3 dir = cosineHemisphere(N, seed);

        // Through the transmission walk, not a binary opaque test, so a
        // translucent flight passes tinted ambient just as it passes tinted
        // direct light.
        vec3 vis = shadowTransmission(origin, dir, radius);
        open += dot(vis, vec3(0.33333));
        // A high mip: this is an irradiance lookup, so it wants the blurred
        // end of the chain rather than the sharp image the mirror uses. Tinted
        // by what survived the flight, so coloured ambient matches the
        // coloured shadow the same flight already casts.
        env += roomEnvDiffuse(
            textureLod(envSampler, dirToEquirect(dir), 5.0).rgb,
            dir, scene.envRoom) * vis;
    }

    ao = open / float(NUM_SAMPLES);
    // Occluded directions contribute nothing, so this already carries the
    // wide-radius occlusion; do not multiply by ao again at the call site.
    irradiance = env / float(NUM_SAMPLES);
}

// Integrate the ring fixture: every sample is a real point on the ring,
// contributing its own direction, distance and visibility. Approximating the
// ring by one on-axis light would make the board centre the brightest point and
// the double ring fall off, which is backwards -- the fixture is mounted out at
// the surround, so illumination is strongest near the rim.
// Diffuse is a colour here, not a scalar, for the same reason it is one on the
// point-light path: a translucent flight tints the light that passes through
// it, so the shadow it casts is coloured.
// Takes F0 rather than a finished Fresnel: the caller's fresnel is built from
// the half-vector to scene.lightPos, which in ring mode is a fictitious
// stand-in position, not the fixture. Each ring point has its own incident
// direction and therefore its own Schlick term, so it is evaluated per sample.
void ringLighting(vec3 P, vec3 N, vec3 V, float roughness, vec3 F0,
                  out vec3 diffOut, out vec3 specOut) {
    float NdotV = max(dot(N, V), 1e-4);
    const int NS = 32;
    float r = scene.lightRing.y;
    float z = scene.lightRing.z;
    vec3 d = vec3(0.0);
    vec3 sp = vec3(0.0);

    float wsum = 0.0;
    for (int i = 0; i < NS; i++) {
        // Even placement with a small per-fragment rotation: a fixed set of 32
        // directions would band on curved surfaces.
        float a = 6.28318530718 * ((float(i) + 0.5) / float(NS)
                                   + sisalHash(gl_FragCoord.xy * 0.61) / float(NS));
        // Accumulated BEFORE any skip, so the normalisation runs over the
        // whole ring and points facing away still dilute the result.
        float w = ringEmission(a);
        wsum += w;
        if (w <= 0.0) continue;
        vec3 Lp = vec3(cos(a) * r, sin(a) * r, z);

        vec3 dl = Lp - P;
        float dist = length(dl);
        vec3 L = dl / dist;
        float ndl = max(dot(N, L), 0.0);
        if (ndl <= 0.0) continue;

        // Transmission walk, so a translucent flight casts a tinted shadow
        // rather than a black one.
        vec3 vis = shadowTransmission(P, L, dist - 0.004);
        vis = mix(vec3(1.0), vis, scene.material.w);

        // The softening constant is the fixture's own size (0.02 BU^2, a ~14mm
        // tube radius), not the 1.0 the point-light path uses. 1.0 is harmless at
        // 20-40 BU, where it shifts the falloff by under a quarter percent, but the
        // ring sits 0.5-1.4 BU off the face at radius 2.35-3.0, so board-to-ring
        // distances run 0.5-5 BU. There 1.0 would flatten the near-field falloff
        // toward uniform (edge-to-centre 2.9x against a true 7.5x for a close
        // ring), and that evenness hides the fibre.
        const float kRingTube2 = 0.02;
        float atten = scene.lightPos.w / (dist * dist + kRingTube2);
        d += w * vis * ndl * atten;
        vec3 Hh = normalize(L + V);
        float vdh = max(dot(V, Hh), 0.0);
        vec3 F = F0 + (1.0 - F0) * pow(1.0 - vdh, 5.0);
        sp += w * vis * specularGGX(F, max(dot(N, Hh), 0.0), NdotV, ndl, roughness)
              * ndl * atten;
    }
    diffOut = d / max(wsum, 1e-4);
    specOut = sp / max(wsum, 1e-4);
}

// `N` is needed because the lobe is not centred on the mirror direction at
// every roughness: a GGX lobe's peak migrates toward the normal as the surface
// roughens, and a fully rough metal reflects a cosine lobe about N, not a
// blurred mirror. `irradiance` is the shading point's own, used to estimate
// how brightly nearby geometry is lit -- see the hit branch.
vec3 traceReflection(vec3 origin, vec3 reflectDir, vec3 N, float roughness,
                     vec3 irradiance) {
    const int NUM_REFL_SAMPLES = 16;
    vec3 result = vec3(0.0);

    // alpha = roughness^2, the same mapping specularGGX uses, so a surface
    // blurs its environment exactly as much as its BRDF blurs the direct
    // highlight. Reflection is a metal's entire environment response, so this
    // mapping largely decides what a barrel looks like.
    float spread = roughness * roughness;

    for (int i = 0; i < NUM_REFL_SAMPLES; i++) {
        vec3 dir;
        if (spread < 0.001) {
            dir = reflectDir;
        } else {
            vec2 seed = gl_FragCoord.xy * 0.71 + vec2(float(i) * 3.91, float(i) * 7.53);
            // Axis migrates mirror -> normal as the surface roughens, so the
            // limits are both right: a mirror at spread 0, a cosine lobe about
            // N at spread 1. Centring every lobe on the mirror direction would
            // make a fully rough metal a blurred mirror, which is not what a
            // rough metal is.
            vec3 axis = normalize(mix(reflectDir, N, spread));
            dir = normalize(mix(axis, cosineHemisphere(axis, seed), spread));
            // Reject directions below the surface. Tested against N, not
            // reflectDir, so a grazing lobe cannot sample into the geometry it
            // sits on.
            if (dot(dir, N) < 0.0) dir = axis;
        }

        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT,
                              0xFF, origin, 0.001, dir, 100.0);
        while (rayQueryProceedEXT(rq)) {}

        if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
            // Hit geometry. The material buffer resolves the instance index
            // to a surface colour, and the board face resolves further to the
            // specific bed at the hit point, which is what puts red and green
            // onto the metal.
            float t = rayQueryGetIntersectionTEXT(rq, true);
            int matIdx = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
            vec3 hitPos = origin + dir * t;
            vec3 hitAlbedo = rayHitAlbedo(matIdx, hitPos);

            // The hit surface is lit too, and how brightly is not known without
            // tracing onward from it. It is approximated by the shading point's
            // OWN irradiance, in the same form the main ambient term uses
            // (albedo * ambient * irradiance): nearby geometry is lit much like
            // the point reflecting it, which is exactly the case this branch
            // handles -- a barrel seeing the board 10-50mm away. Distance
            // falloff is kept: a bed 10mm from the barrel bleeds far more onto
            // it than the far side of the board does.
            result += hitAlbedo * scene.cameraPos.w * irradiance
                      / (1.0 + t * t);
        } else {
            // Escaped to the environment, the common case here. Without it
            // metals have nothing to reflect and read as plastic.
            // Widened with roughness, matching the ray jitter above: a
            // brushed barrel smears the fixture where a polished one
            // mirrors it.
            result += roomEnvSpecular(
                textureLod(envSampler, dirToEquirect(dir),
                           roughness * 6.0).rgb,
                dir, scene.envRoom, roughness * roughness * 60.0);
        }
    }
    return result / float(NUM_REFL_SAMPLES);
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------
void main() {
    // Decode packed draw mode
    bool useTex      = (fragDrawMode & (1u << 31)) != 0u;
    bool hasNormalMap = (fragDrawMode & (1u << 30)) != 0u;
    float metallic  = float((fragDrawMode >> 15) & 0xFFu) / 255.0;
    float roughness = float((fragDrawMode >> 7)  & 0xFFu) / 255.0;
    float sisalOcc = 1.0;

    bool isBoardFace = (fragDrawMode & (1u << 29)) != 0u;

    vec3 albedo;
    if (isBoardFace) {
        // Evaluated, not sampled -- and through the SAME function the ray path
        // uses, so a bed cannot be one colour to the camera and another to a
        // reflection.
        //
        // Board-local rather than world XY: the board carries a mounting
        // rotation and a face rotation, and the beds turn with them.
        vec3 local = (scene.boardFaceInvModel * vec4(fragWorldPos, 1.0)).xyz;
        albedo = boardFaceAlbedo(local.xy, scene.boardPalette);
    } else if (useTex) {
        albedo = texture(texSampler, fragUV).rgb * fragColor;
    } else {
        albedo = fragColor;
    }

    if (isBoardFace) {
        albedo = applyDecals(albedo, fragWorldPos.xy);
    }

    vec3 N = normalize(fragNormal);

    // Apply normal map if present
    if (hasNormalMap) {
        vec3 T = normalize(fragTangent);
        T = normalize(T - dot(T, N) * N); // re-orthogonalize
        vec3 B = cross(N, T) * fragBitangentSign;
        mat3 TBN = mat3(T, B, N);

        vec3 mapNormal = texture(normalSampler, fragUV).rgb * 2.0 - 1.0;
        // Strength is material.x.
        mapNormal = mix(vec3(0.0, 0.0, 1.0), mapNormal, scene.material.x);
        N = normalize(TBN * mapNormal);
    }

    // Sisal fibre and wear. The footprint is measured out here rather than
    // inside the branch because derivatives are only defined in uniform control
    // flow, and only the board face takes this path.
    vec2 sisalQ = fragWorldPos.xy * scene.sisal.x;
    float sisalFootprint = max(length(dFdx(sisalQ)), length(dFdy(sisalQ)));
    if (isBoardFace) {
        sisalOcc = sisalSurface(albedo, N, roughness, fragWorldPos.xy, sisalFootprint);
    }

    vec3 L = normalize(scene.lightPos.xyz - fragWorldPos);
    vec3 V = normalize(scene.cameraPos.xyz - fragWorldPos);
    vec3 H = normalize(L + V);

    // Ambient with AO
    float ambient = scene.cameraPos.w;
    // Two radii: the wide term is broad environmental darkening, the tight one
    // is the contact seam where a dart enters the board. 0.3 BU is ~30mm
    // against a 225mm board radius, far too coarse to draw that seam, and its
    // absence is what leaves darts looking pasted onto the surface rather
    // than seated in it.
    //
    // The wide trace also returns the environment radiance it saw, so ambient
    // is directional and coloured at no extra ray cost.
    float aoWide;
    vec3  irradiance;
    traceAmbient(fragWorldPos + N * 0.005, N, 0.3, aoWide, irradiance);
    float aoTight = traceAO(fragWorldPos + N * 0.002, N, 0.03);
    float ao = aoWide * mix(1.0, aoTight, scene.material.z);

    // Fresnel (Schlick approximation). Schlick takes the view-to-half angle;
    // dot(N, H) would tie Fresnel to the specular lobe rather than to viewing
    // angle and remove grazing-angle rim brightening.
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float VdotH = max(dot(V, H), 0.0);
    vec3 fresnel = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

    // Direct lighting, accumulated with attenuation and visibility folded in so
    // the two fixture types can be integrated differently without the rest of
    // the shader caring which one is active.
    // Diffuse is a colour, not a scalar: a translucent flight tints the light
    // it passes, so the shadow it casts is coloured.
    vec3  lightDiff;
    vec3  lightSpec;
    float NdotV = max(dot(N, V), 1e-4);
    if (scene.lightRing.x > 0.5) {
        vec3 ringDiff;
        ringLighting(fragWorldPos + N * 0.005, N, V, roughness, F0,
                     ringDiff, lightSpec);
        lightDiff = ringDiff;
    } else {
        float dist = length(scene.lightPos.xyz - fragWorldPos);
        float atten = scene.lightPos.w / (dist * dist + 1.0);
        // Coloured, because a translucent flight tints what passes through it.
        vec3 rawShadow = traceShadow(fragWorldPos + N * 0.005, scene.lightPos.xyz, 40);
        vec3 shadow = mix(vec3(1.0), rawShadow, scene.material.w);
        float NdotH = max(dot(N, H), 0.0);
        float NdotL = max(dot(N, L), 0.0);
        lightDiff = NdotL * atten * shadow;
        // Specular is weighted by NdotL like any other radiance term, so a
        // highlight goes to zero on surfaces turned away from the light.
        lightSpec = specularGGX(fresnel, NdotH, NdotV, NdotL, roughness)
                    * NdotL * atten * shadow;
    }

    // Second fixture. Positioned, so unlike the environment probe it produces
    // an actual gradient across the board -- and a second shadow direction,
    // which is what stops the lighting reading as a single studio lamp.
    vec3 light2 = vec3(0.0);
    if (scene.light2Pos.w > 0.0) {
        vec3 dl2 = scene.light2Pos.xyz - fragWorldPos;
        float dist2 = length(dl2);
        vec3 L2 = dl2 / max(dist2, 1e-4);
        float NdotL2 = max(dot(N, L2), 0.0);
        if (NdotL2 > 0.0) {
            float atten2 = scene.light2Pos.w / (dist2 * dist2 + 1.0);
            vec3 shadow2 = mix(vec3(1.0),
                               traceShadow(fragWorldPos + N * 0.005,
                                           scene.light2Pos.xyz, 24),
                               scene.material.w);
            vec3 H2 = normalize(L2 + V);
            // Its own Fresnel: the key light's was built from a different
            // half-vector.
            vec3 F2 = F0 + (1.0 - F0) * pow(1.0 - max(dot(V, H2), 0.0), 5.0);
            vec3 d2 = albedo * NdotL2 * (1.0 - metallic);
            vec3 s2 = specularGGX(F2, max(dot(N, H2), 0.0), NdotV, NdotL2,
                                  roughness) * NdotL2;
            light2 = (d2 + s2) * atten2 * shadow2 * scene.light2Color.rgb;
        }
    }

    // irradiance already carries the wide-radius occlusion (occluded rays
    // returned nothing), so only the tight contact term is applied on top of
    // it; multiplying by the full `ao` would count the wide term twice.
    //
    // Ambient DIFFUSE, so metals are excluded exactly as they are from the
    // direct diffuse on the next line. A metal has no diffuse reflection; its
    // environment response is the reflection term below. Without (1 - metallic)
    // every metal takes a view-independent Lambertian wash (`ambient` runs as
    // high as 4.2) and dart barrels come out uniformly near-white.
    vec3 color = albedo * ambient * irradiance * (1.0 - metallic)
                 * mix(1.0, aoTight, scene.material.z)
               + (albedo * lightDiff * (1.0 - metallic) + lightSpec)
                 * scene.lightColor.rgb
               + light2;

    // Reflection. Applied to dielectrics too, at reduced weight: a clear
    // coat reflecting its surroundings is part of why a real object sits in
    // a scene, and without it non-metals have a purely local response.
    vec3 refl = traceReflection(fragWorldPos + N * 0.005, reflect(-V, N), N,
                                roughness, irradiance);
    vec3 F_env = F0 + (max(vec3(1.0 - roughness), F0) - F0)
                      * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    // No AO factor here. traceReflection resolves visibility exactly -- a ray
    // that would be occluded HITS the occluder and returns its colour, and a
    // ray that escapes was demonstrably not occluded -- so multiplying the
    // result by the ambient rays' occlusion would count the same blockers twice.
    color += refl * F_env * scene.material.y * mix(0.25, 1.0, metallic);

    // Fibre self-shadowing, applied to the assembled radiance so it occludes
    // the specular and reflected terms too, not just the diffuse one -- and
    // pre-compensated for the tone curve's shoulder.
    //
    // Scaling radiance is the physically honest operation, but it is not what
    // the eye gets: a modulation that reads 15% deep in shadow arrives at under
    // 2% anywhere the face is brightly lit, because the shoulder's log-log
    // slope goes to zero. No reshaping of the field in linear space avoids
    // that; shallow and sparse-but-deep fields collapse the same way.
    //
    // So the occlusion is raised to the inverse of the curve's local slope,
    // which is a log-space scaling: what survives to the output is then roughly
    // the same depth regardless of exposure. This is a deliberate departure
    // from pure radiance scaling, in the same spirit as the local tone mapping
    // a real camera applies to hold micro-contrast through a highlight. Clamped
    // so dark regions are untouched and the boost cannot run away where the
    // curve is flattest.
    if (sisalOcc < 0.999) {
        float L  = max(dot(color, vec3(0.2126, 0.7152, 0.0722)), 1e-4);
        float a0 = max(tonemapACES1(L), 1e-4);
        float slope = (tonemapACES1(L * 1.05) - a0) / (a0 * 0.05);
        color *= pow(sisalOcc, clamp(1.0 / max(slope, 1e-3), 1.0, 6.0));
    }
    color = tonemapACES(color);

    // Translucent dart flights carry their alpha in the low seven bits of
    // drawMode; everything else is opaque.
    float outAlpha = 1.0;
    if ((fragDrawMode & (1u << 28)) != 0u)
        outAlpha = float(fragDrawMode & 0x7Fu) / 127.0;
    outColor = vec4(color, outAlpha);
}
