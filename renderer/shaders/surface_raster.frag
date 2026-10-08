#version 450
#extension GL_GOOGLE_include_directive : require

// Set layout: 0 UBO, 1 albedo, 2 normal map, 3 environment, 4 decal atlas.
#define DECAL_SET 4

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

// Environment: the same background image composited behind the scene, reused
// as a crude reflection probe.  A metal is mostly a mirror, so with no
// environment to reflect it collapses to flat ambient and reads as plastic.
layout(set = 3, binding = 0) uniform sampler2D envSampler;

void main() {
    bool useTex       = (fragDrawMode & (1u << 31)) != 0u;
    bool hasNormalMap = (fragDrawMode & (1u << 30)) != 0u;
    float metallic  = float((fragDrawMode >> 15) & 0xFFu) / 255.0;
    float roughness = float((fragDrawMode >> 7)  & 0xFFu) / 255.0;
    float sisalOcc = 1.0;

    bool isBoardFace = (fragDrawMode & (1u << 29)) != 0u;
    // The face is evaluated, not sampled, and through the same shared function
    // the RT path uses -- see board_face.glsl.
    vec3 albedo;
    if (isBoardFace) {
        vec3 local = (scene.boardFaceInvModel * vec4(fragWorldPos, 1.0)).xyz;
        albedo = boardFaceAlbedo(local.xy, scene.boardPalette);
    } else {
        albedo = useTex ? texture(texSampler, fragUV).rgb * fragColor : fragColor;
    }

    // Board face only: world XY is board XY up to the board's own Z rotation,
    // and since placement is re-randomised every frame that distinction is not
    // observable.
    if (isBoardFace) {
        albedo = applyDecals(albedo, fragWorldPos.xy);
    }

    vec3 N = normalize(fragNormal);

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

    float ambient = scene.cameraPos.w;
    float diff = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 1e-4);

    float dist = length(scene.lightPos.xyz - fragWorldPos);
    float atten = scene.lightPos.w / (dist * dist + 1.0);

    // Fresnel (Schlick approximation). Schlick takes the view-to-half angle;
    // dot(N, H) would tie Fresnel to the specular lobe rather than to viewing
    // angle and remove grazing-angle rim brightening.
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float VdotH = max(dot(V, H), 0.0);
    vec3 fresnel = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

    float NdotH = max(dot(N, H), 0.0);
    // Specular is weighted by NdotL like any other radiance term, so a
    // highlight goes to zero on surfaces turned away from the light.
    vec3 spec = specularGGX(fresnel, NdotH, NdotV, diff, roughness) * diff;

    vec3 diffContrib = albedo * diff * (1.0 - metallic);

    // Ring fixture: integrate over the ring rather than treating it as one
    // on-axis light, which would make the board centre brightest and the double
    // ring fall off -- backwards for a fixture mounted out at the surround.
    if (scene.lightRing.x > 0.5) {
        const int NS = 24;
        float rr = scene.lightRing.y;
        float zz = scene.lightRing.z;
        float d = 0.0;
        vec3 sp = vec3(0.0);
        // Emission profile (ringEmission): real board lights are horseshoes
        // or unevenly lit rings, and a perfect circle lights the face
        // unrealistically evenly.
        float wsum = 0.0;
        for (int i = 0; i < NS; i++) {
            float a = 6.28318530718 * (float(i) + 0.5) / float(NS);
            float w = ringEmission(a);
            wsum += w;
            if (w <= 0.0) continue;
            vec3 dl = vec3(cos(a) * rr, sin(a) * rr, zz) - fragWorldPos;
            float dd = length(dl);
            vec3 Lr = dl / dd;
            float ndl = max(dot(N, Lr), 0.0);
            if (ndl <= 0.0) continue;
            // The softening constant is the fixture's own size (0.02 BU^2, a ~14mm
            // tube radius), not the 1.0 the point-light path uses. 1.0 is harmless at
            // 20-40 BU, where it shifts the falloff by under a quarter percent, but the
            // ring sits 0.5-1.4 BU off the face at radius 2.35-3.0, so board-to-ring
            // distances run 0.5-5 BU. There 1.0 would flatten the near-field falloff
            // toward uniform (edge-to-centre 2.9x against a true 7.5x for a close
            // ring), and that evenness hides the fibre.
            const float kRingTube2 = 0.02;
            float at = scene.lightPos.w / (dd * dd + kRingTube2);
            d += w * ndl * at;
            vec3 Hr = normalize(Lr + V);
            // Per-sample Schlick: the shared `fresnel` above is built from the
            // half-vector to scene.lightPos, a fictitious stand-in in ring
            // mode. Each ring point gets its own term.
            float vdh = max(dot(V, Hr), 0.0);
            vec3 Fr = F0 + (1.0 - F0) * pow(1.0 - vdh, 5.0);
            sp += w * specularGGX(Fr, max(dot(N, Hr), 0.0), NdotV, ndl, roughness)
                  * ndl * at;
        }
        diff = d / max(wsum, 1e-4);
        spec = sp / max(wsum, 1e-4);
        diffContrib = albedo * diff * (1.0 - metallic);
        atten = 1.0;
    }

    // Ambient from the environment rather than a flat scalar.
    //
    // A blurred probe lookup along the normal is the cheap stand-in for a
    // diffuse irradiance map: it costs one texture fetch and gives ambient a
    // direction and a colour. A neutral constant, against a key light ranging
    // over 3000-7000K, would be a grey fill no light in the scene casts --
    // flattening the shading and pulling everything toward the same tint.
    //
    // The RT shader gathers this properly from its AO rays; this path has no
    // rays to gather from, so it approximates with the normal-direction lookup.
    vec3 irradiance = roomEnvDiffuse(
        textureLod(envSampler, dirToEquirect(N), 6.0).rgb, N, scene.envRoom);

    // Second fixture. No shadow ray on this path -- there are none to cast --
    // but the gradient and the cross-temperature shading are most of what it
    // contributes, and both survive without one.
    vec3 light2 = vec3(0.0);
    if (scene.light2Pos.w > 0.0) {
        vec3 dl2 = scene.light2Pos.xyz - fragWorldPos;
        float dist2 = length(dl2);
        vec3 L2 = dl2 / max(dist2, 1e-4);
        float NdotL2 = max(dot(N, L2), 0.0);
        if (NdotL2 > 0.0) {
            float atten2 = scene.light2Pos.w / (dist2 * dist2 + 1.0);
            vec3 H2 = normalize(L2 + V);
            vec3 F2 = F0 + (1.0 - F0) * pow(1.0 - max(dot(V, H2), 0.0), 5.0);
            light2 = (albedo * NdotL2 * (1.0 - metallic)
                      + specularGGX(F2, max(dot(N, H2), 0.0), NdotV, NdotL2,
                                    roughness) * NdotL2)
                     * atten2 * scene.light2Color.rgb;
        }
    }

    // (1 - metallic) because this is the ambient DIFFUSE term and metals have
    // no diffuse reflection -- see the matching note in surface_rt.frag.
    vec3 color = albedo * ambient * irradiance * (1.0 - metallic)
               + (diffContrib + spec) * atten * scene.lightColor.rgb
               + light2;

    // Environment reflection. Without this a metallic surface loses its
    // diffuse term to (1 - metallic) and gets nothing back, ending up darker
    // and flatter than the same surface left dielectric.
    vec3 R = reflect(-V, N);
    vec3 envCol = roomEnvSpecular(
        textureLod(envSampler, dirToEquirect(R), roughness * 6.0).rgb,
        R, scene.envRoom, roughness * roughness * 60.0);
    vec3 F_env = F0 + (max(vec3(1.0 - roughness), F0) - F0)
                      * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    color += envCol * F_env * scene.material.y * mix(0.25, 1.0, metallic);

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
