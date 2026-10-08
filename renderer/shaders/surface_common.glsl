// Shading shared by surface_raster.frag and surface_rt.frag: the scene UBO,
// printed-mark decals, sisal fibre and wear, the ring fixture's emission
// profile, tone mapping and the GGX specular lobe.
//
// One copy, so the ray-traced path and the raster fallback cannot drift apart
// in anything but what genuinely differs between them -- how visibility,
// ambient and reflections are gathered.
//
// The including shader defines DECAL_SET, the descriptor set of the decal
// atlas: it sits after the environment probe, whose set number depends on
// whether the TLAS set is present.

#ifndef SURFACE_COMMON_GLSL
#define SURFACE_COMMON_GLSL

#ifndef DECAL_SET
#error "define DECAL_SET before including surface_common.glsl"
#endif

// Scene UBO. Layout must match SceneUBO in renderer/src/render_pass.h, whose
// static_assert pins the size.
layout(set = 0, binding = 0) uniform SceneUBO {
    vec4 lightPos;       // xyz = position, w = intensity
    vec4 cameraPos;      // xyz = position, w = ambient strength
    vec4 lightColor;     // rgb = color, w = unused
    vec4 material;       // x = normal strength, y = env intensity,
                         // z = contact AO strength, w = shadow strength
    vec4 lightRing;      // x = mode (0 point, 1 ring), y = ring radius,
                         // z = ring offset along the board normal
    vec4 decal[30];      // 5 slots x 6: see applyDecals below
    vec4 decalGrid;      // xy = glyph columns, font rows
    vec4 sisal;          // x = fibre cells per board unit, y = fibre contrast,
                         // z = fibre relief, w = global wear amount
    vec4 wear[8];        // xy = centre (board units), z = radius, w = weight
    /// Board-local frame, so the face can be evaluated at a fragment or a
    /// ray hit.
    mat4 boardFaceInvModel;
    /// The room's bright end: rgb = fixture colour, w = its radiance. Well
    /// above 1 -- see env_room.glsl for why the photo alone cannot supply it.
    vec4 envRoom;
    /// Bed and ring colours for this frame, linear; columns are black bed,
    /// cream bed, ring-on-black, ring-on-cream. See board_face.glsl.
    mat4 boardPalette;
    /// Second fixture: xyz = position, w = intensity (0 disables).
    vec4 light2Pos;
    vec4 light2Color;
    /// Ring emission profile: gap centre, gap half-width, asymmetry amplitude,
    /// asymmetry phase. All angles in radians. See ringEmission.
    vec4 lightRingProfile;
} scene;

// Decal atlas: glyph sheet, one row per font, one column per character.
layout(set = DECAL_SET, binding = 0) uniform sampler2D decalSampler;

// Procedurally assembled printed marks. Real boards carry logos and printing,
// and a network trained on a board without them mistakes them for darts, so a
// few marks are composited onto the face each frame at random placement and
// never annotated.
//
// The atlas (tools/generate_decal_atlas.py) holds GLYPHS, one row per font and
// one column per character, and a mark is assembled here from a random glyph
// sequence so that no mark repeats. A fixed set of finished wordmarks would
// each be seen tens of thousands of times over a training run, and the network
// would learn those specific marks instead of the class.
//
// Applied only where bit 29 of drawMode is set, i.e. the board face -- not the
// wires, numerals or darts.
const int DECAL_SLOTS = 5;

float glyphColumn(vec4 lo, vec4 hi, int i) {
    vec4 v = (i < 4) ? lo : hi;
    int j = i & 3;
    return (j == 0) ? v.x : ((j == 1) ? v.y : ((j == 2) ? v.z : v.w));
}

vec3 applyDecals(vec3 albedo, vec2 boardXY) {
    vec2 atlas = scene.decalGrid.xy;   // glyph columns, font rows
    for (int i = 0; i < DECAL_SLOTS; ++i) {
        vec4 A = scene.decal[i * 6 + 0];  // fontRow, wordLen, cos, sin
        vec4 B = scene.decal[i * 6 + 1];  // centre.xy, halfW, halfH
        vec4 C = scene.decal[i * 6 + 2];  // tint.rgb, opacity
        vec4 D = scene.decal[i * 6 + 3];  // glyph columns 0..3
        vec4 E = scene.decal[i * 6 + 4];  // glyph columns 4..7
        vec4 F = scene.decal[i * 6 + 5];  // style, shape, curvature, stroke

        if (C.a <= 0.0 || B.z <= 0.0 || B.w <= 0.0) continue;

        vec2 rel = boardXY - B.xy;
        // Into the mark's own frame: +x runs along the baseline, +y radially.
        vec2 local = vec2(rel.x * A.z + rel.y * A.w,
                         -rel.x * A.w + rel.y * A.z) / B.zw;
        if (abs(local.x) > 1.0 || abs(local.y) > 1.0) continue;

        float style  = F.x;   // 0 plain, 1 outlined badge, 2 knockout patch
        float shape  = F.y;   // 0 rounded rect, 1 ellipse
        float curv   = F.z;   // baseline bend, following the rim
        float stroke = F.w;

        // Text sits inside the badge when there is one.
        float inset = (style > 0.5) ? 0.62 : 0.94;
        vec2 tp = local / inset;
        // Bend the baseline so the mark follows the annulus rather than
        // cutting across it as a chord.
        tp.y += curv * (tp.x * tp.x - 0.3333);

        float ink = 0.0;
        if (abs(tp.x) <= 1.0 && abs(tp.y) <= 1.0) {
            float L = max(A.y, 1.0);
            float u = (tp.x * 0.5 + 0.5) * L;
            int slot = int(floor(u));
            if (slot >= 0 && slot < int(L)) {
                vec2 cell = vec2(glyphColumn(D, E, slot), A.x);
                vec2 g = vec2(fract(u), tp.y * 0.5 + 0.5);
                ink = texture(decalSampler, (g + cell) / atlas).a;
            }
        }

        float cover;
        if (style < 0.5) {
            cover = ink;
        } else {
            float sd = (shape < 0.5)
                ? max(abs(local.x) / 0.97, abs(local.y) / 0.92)
                : length(local / vec2(0.97, 0.92));
            float inside = 1.0 - smoothstep(0.98, 1.0, sd);
            if (style < 1.5) {
                float inner = 1.0 - smoothstep(1.0 - stroke - 0.02,
                                               1.0 - stroke, sd);
                cover = max(ink, inside - inner);
            } else {
                // Knockout: the glyphs are unprinted, so the ink's own alpha is
                // punched out and the board shows through the lettering.
                cover = inside * (1.0 - ink);
            }
        }
        albedo = mix(albedo, C.rgb, clamp(cover, 0.0, 1.0) * C.a);
    }
    return albedo;
}

// ---------------------------------------------------------------------------
// Sisal fibre and wear (board face only)
// ---------------------------------------------------------------------------
//
// board_face.glsl supplies flat colour with no relief, which reads as printed
// card rather than as the cut ends of tightly packed sisal fibre. Fibre, wear
// and dart holes are added here.
//
// Generated per frame rather than baked into a texture because wear has to
// differ from board to board. A fixed wear pattern is a fixed image feature,
// and over a few hundred epochs the network learns that particular board
// instead of learning that boards are worn -- the same reason printed marks
// are assembled per frame rather than stored.

float sisalHash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float sisalNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(sisalHash(i),                  sisalHash(i + vec2(1.0, 0.0)), u.x),
               mix(sisalHash(i + vec2(0.0, 1.0)), sisalHash(i + vec2(1.0, 1.0)), u.x), u.y);
}

// Footprint-aware fBm: each octave fades out as its cell size drops below the
// pixel footprint.
//
// Sisal is the highest-frequency content on the board. Drawn without this, the
// finest octaves alias into per-frame shimmer -- a synthetic-only signature
// sitting on exactly the thin features the board head localises. The 2x
// supersample shades about 0.55mm per pixel, so with the base octave near
// 3.5mm roughly three octaves survive at a typical framing and the rest drop
// out on their own as the camera pulls back.
float sisalFbm(vec2 p, float footprint) {
    float sum = 0.0, total = 0.0, amp = 1.0, freq = 1.0;
    for (int i = 0; i < 5; ++i) {
        // Fade an octave out as its cell size approaches the pixel footprint.
        // The constant is 1.2 rather than a strict Nyquist 2.0 because the
        // frame is shaded at 2x and box-filtered down: detail at about one
        // shaded pixel per cell is resolved by the supersample and averaged by
        // the filter, which is what a real sensor does with it too.
        float fade = clamp(1.0 - footprint * freq * 1.2, 0.0, 1.0);
        if (fade > 0.0) sum += amp * fade * (sisalNoise(p * freq) - 0.5);
        total += amp;      // full weight, faded or not
        amp   *= 0.5;
        freq  *= 2.0;
    }
    // Normalising by the FULL weight rather than the faded weight is what makes
    // contrast decay smoothly to flat as the camera pulls back. Dividing by the
    // faded weight instead holds contrast at full until the last octave dies
    // and then snaps to flat, which pops between frames.
    return 0.5 + sum / total;
}

// Packed fibre ends.
//
// Value-noise fBm alone reads as cloud or plaster: it is smooth everywhere,
// and sisal is not. A bristle board is the cut end of tightly packed fibre
// bundles, so the structure is cellular -- small convex ends with darker gaps
// between them. This is a Worley F1 field at bundle scale, which is what gives
// the surface its grain rather than a wash.
//
// Footprint-faded like the fBm, for the same reason: it is high-frequency
// content and unfiltered it would alias into per-frame shimmer.
float sisalCells(vec2 p, float footprint, float freq) {
    float fade = clamp(1.0 - footprint * freq * 1.2, 0.0, 1.0);
    if (fade <= 0.0) return 0.5;

    vec2 q = p * freq;
    vec2 base = floor(q);
    vec2 f = q - base;
    float d1 = 8.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec2 g = vec2(float(i), float(j));
            // Jittered site per cell; the jitter is what stops the bundles
            // lining up into a visible lattice.
            vec2 site = g + vec2(sisalHash(base + g),
                                 sisalHash(base + g + vec2(37.0, 17.0)));
            float d = dot(site - f, site - f);
            d1 = min(d1, d);
        }
    }
    // sqrt gives distance; the curve pushes most of the cell bright and keeps
    // the darkening to the seams, which is where a fibre gap actually is.
    float v = 1.0 - clamp(sqrt(d1) * 1.35, 0.0, 1.0);
    return mix(0.5, v, fade);
}

// Accumulated wear. The hotspots are placed by the randomizer on the same aim
// targets placeDarts throws at, so wear lands where this renderer's darts
// actually cluster. Slot count must match FrameState::kMaxWearSpots.
float sisalWear(vec2 bp) {
    float w = 0.0;
    for (int i = 0; i < 8; ++i) {
        vec4 h = scene.wear[i];
        if (h.w <= 0.0) continue;
        vec2 d = (bp - h.xy) / max(h.z, 1e-4);
        w += h.w * exp(-dot(d, d));
    }
    return clamp(w, 0.0, 1.0);
}

// Dart holes. One candidate per cell, present only in proportion to wear, so a
// lightly used board shows a few and a hammered treble twenty is peppered.
//
// Sisal closes over a hole rather than leaving it open, so these read as dark
// pits, not punctures. At the caller's frequency a cell is roughly 4mm, which
// the 2x supersample resolves at ~7 shaded pixels across -- big enough to
// survive the box filter down to the output image.
float sisalPocks(vec2 p, float density) {
    vec2 ip = floor(p), fp = fract(p);
    float acc = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 g = vec2(float(x), float(y));
            vec2 c = ip + g;
            // Wear drives how MANY holes there are, not how deep they are: a
            // lightly used bed has a few full-depth holes, not a uniform film
            // of shallow ones.
            float present = step(sisalHash(c + vec2(5.0, 11.0)), density);
            vec2 o = vec2(sisalHash(c), sisalHash(c + vec2(17.0, 3.0)));
            float d = length(fp - g - o);
            acc = max(acc, present * (1.0 - smoothstep(0.10, 0.34, d)));
        }
    }
    return acc;
}

// `footprint` is measured by the caller in uniform control flow: derivatives
// taken inside the board-face branch would be undefined.
// Returns a micro-occlusion factor for the fibre structure. The caller must
// apply it to the FINAL radiance, not to albedo: self-shadowing between fibre
// ends occludes specular and reflected light exactly as it occludes diffuse,
// and folded into albedo alone it would be washed out by the additive terms
// wherever the board is brightly lit.
float sisalSurface(inout vec3 albedo, inout vec3 N, inout float roughness,
                  vec2 bp, float footprint) {
    float contrast = scene.sisal.y;
    float relief   = scene.sisal.z;
    float wearAmt  = scene.sisal.w;
    if (scene.sisal.x <= 0.0) return 1.0;

    vec2 q = bp * scene.sisal.x;
    float h0 = sisalFbm(q, footprint);

    // Cellular grain at bundle scale, and a tone speckle that is deliberately
    // NOT the height field.
    //
    // Driving albedo from h0 alone would make tone and relief perfectly
    // correlated, so every bright patch would also be a raised one. Real fibre
    // ends vary in colour independently of how proud they sit -- some bundles
    // are simply paler -- and decorrelating the two is most of what stops the
    // surface reading as embossed paper.
    float cells   = sisalCells(bp, footprint, scene.sisal.x * 0.55);

    // Two tone fields, at deliberately different scales.
    //
    // fibreScale (sisal.x) is 55-85 cells per board unit, i.e. 1.2-1.8mm per
    // cell. At the 1024 output the network trains on, the board spans roughly
    // 1.5 px/mm, so a fibre cell is 2-3 pixels: at the resolution limit, where
    // the footprint fade attenuates it as far as is needed to stop it aliasing.
    //
    // Coarse structure survives at any output size, and it is what makes a
    // real board read as textured: bundles group into patches of slightly
    // different tone several millimetres across. So the coarse mottle carries
    // most of the tonal variation and the fine speckle is a detail layer for
    // the resolutions that can resolve it.
    float mottle  = sisalFbm(q * 0.14 + vec2(53.7, 11.3), footprint);   // ~9-13mm
    float speckle = sisalFbm(q * 0.37 + vec2(19.1, 71.5), footprint);   // ~3-5mm

    // One-sided: the gaps between fibre ends are self-shadowing, and a fibre
    // end cannot be brighter than the paint sitting on it.
    //
    // Split by what each term physically is. The cellular/fBm field is
    // geometry -- fibre ends shadowing the gaps between them -- so it occludes
    // every incoming path and is returned for the caller to apply to the final
    // radiance. Mottle and speckle are pigment variation between bundles, so
    // they stay on albedo.
    float shade = mix(cells, h0, 0.35);                  // grain, mostly cellular

    float occ   = mix(1.0 - contrast, 1.0, shade);
    albedo *= mix(1.0 - contrast * 0.85, 1.0, mottle);   // coarse tone
    albedo *= mix(1.0 - contrast * 0.35, 1.0, speckle);  // detail layer

    // Put the grain in the roughness too, not just the albedo.
    //
    // Albedo modulation alone only survives where nothing is bright: in a lit
    // region the tonemapper's shoulder compresses the result toward white and a
    // 22% albedo swing comes out as a couple of percent, while the specular
    // term -- additive and independent of albedo -- paints uniform sheen over
    // whatever is left. Under a bright light the albedo grain all but vanishes.
    //
    // Roughness is the physically right channel for it anyway. A compressed
    // fibre end takes paint and holds a sheen; the seam beside it is deep,
    // fibrous and scatters. Varying it means the highlight itself carries the
    // grain, so the pattern shows up precisely where the albedo grain cannot.
    //
    // Modulated DOWNWARD from the authored value, never up: the face is
    // authored near the ceiling for matte sisal, so an additive term would be
    // clamped away. It is the fibre ends that gloss, not the gaps.
    roughness = clamp(roughness - shade * contrast * 1.5
                                + mottle * contrast * 0.30, 0.15, 1.0);

    float wear = sisalWear(bp) * wearAmt;
    if (wear > 0.0) {
        // Paint comes off in patches rather than evenly. It survives in the
        // hollows between fibre ends and goes first off the tips the points
        // keep striking, so a worn bed is blotchy -- fading the whole bed
        // uniformly toward tan just reads as a faded photograph. The patches
        // run at ~30mm, well above the fibre itself.
        float bare = sisalNoise(bp * 3.3 + vec2(9.1, 4.7));   // ~30mm patches
        float strip = clamp(wear * smoothstep(0.62 - wear * 0.50,
                                              1.00 - wear * 0.40, bare),
                            0.0, 1.0);
        // Tan against black is a large change and tan against cream is a small
        // one, which is why a worn board shows its wear mostly on the black
        // beds. That falls out of mixing toward the fibre colour.
        const vec3 rawSisal = vec3(0.74, 0.66, 0.50);
        albedo = mix(albedo, rawSisal * mix(0.72, 1.0, h0), strip * 0.50);

        // Dart holes; see sisalPocks.
        albedo *= 1.0 - sisalPocks(bp * 25.0, wear) * 0.45;   // ~4mm cells

        relief *= 1.0 + wear * 1.5;   // worn fibre stands up
    }

    // Relief from the height field. The face is planar with its normal along
    // +Z, so the gradient in board xy is already the world-space gradient and
    // no tangent frame is needed, which also avoids the seam the mesh tangents
    // carry.
    // Relief follows the same combined field the shading does, so a bundle
    // that reads bright also reads convex.
    float e = 1.0;
    float hx = mix(sisalCells(bp + vec2(e / (scene.sisal.x * 0.55), 0.0), footprint,
                              scene.sisal.x * 0.55),
                   sisalFbm(q + vec2(e, 0.0), footprint), 0.35);
    float hy = mix(sisalCells(bp + vec2(0.0, e / (scene.sisal.x * 0.55)), footprint,
                              scene.sisal.x * 0.55),
                   sisalFbm(q + vec2(0.0, e), footprint), 0.35);
    N = normalize(N - vec3(hx - shade, hy - shade, 0.0) * relief);

    return occ;
}

// Map a direction to the background image, poled on world UP (+Y) -- the same
// axis as env_room.glsl's floor-to-ceiling gradient. The photo is not a
// panorama, so no mapping is physically correct; this one gives metal
// something varied and scene-correlated to reflect, and agrees with the room
// model about which way is up.
vec2 dirToEquirect(vec3 d) {
    float u = atan(d.z, d.x) * 0.15915494309 + 0.5;   // 1/(2*pi)
    float v = acos(clamp(d.y, -1.0, 1.0)) * 0.31830988618;  // 1/pi
    return vec2(u, v);
}

// Scalar form of the same curve, for measuring its local slope.
float tonemapACES1(float x) {
    return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

// Narkowicz ACES approximation. Reinhard compresses toward white and washes
// out saturated highlights, which is a large part of what makes renders read
// as synthetic next to photographs.
//
// The result stays linear. The colour attachment is SRGB (kColorFormat in
// render_pass.h), which encodes on store and decodes wherever values are
// combined, so blending, the MSAA resolve and the supersample filter all
// operate on linear values.
vec3 tonemapACES(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// --------------------------------------------------------------------------
// Microfacet specular (GGX + height-correlated Smith)
// --------------------------------------------------------------------------
//
// Blinn-Phong's pow(NdotH, shininess) has no geometry term and no
// normalization: its lobe does not conserve energy across roughness, and it
// misses the grazing-angle shadowing that gives metal its characteristic
// falloff.
const float PI = 3.14159265359;

float D_GGX(float NdotH, float a) {
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Height-correlated Smith visibility. This already carries the microfacet
// BRDF's 1/(4 NdotL NdotV) denominator, so it is not applied separately.
float V_SmithGGX(float NdotV, float NdotL, float a) {
    float a2 = a * a;
    float lv = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float ll = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(lv + ll, 1e-5);
}

// The shader's diffuse term is albedo * NdotL rather than albedo/PI * NdotL, so
// the specular lobe is scaled by PI to cancel the one inside D_GGX and keep the
// two terms in consistent relative weight. The randomizer's light intensity
// ranges are calibrated against this convention.
vec3 specularGGX(vec3 F, float NdotH, float NdotV, float NdotL, float rough) {
    float a = max(rough * rough, 1e-3);
    return F * (D_GGX(NdotH, a) * V_SmithGGX(NdotV, NdotL, a) * PI);
}

// Angular emission weight of the ring fixture, at angle `a` around it.
//
// A perfect uniform circle is the one thing a real board light is not. Many are
// horseshoes with a break at the mount or the cable entry, and an LED ring has
// hot and dull segments around its circumference -- often a dead one. A
// uniform ring lights the face perfectly evenly, and that evenness is
// geometric: no lamp distance or amount of fill removes it.
//
// Callers normalise by the SUM of these weights rather than by the sample
// count, so a gap redistributes the fixture's output instead of dimming it.
// A fixture is specified by total lumens; what a horseshoe changes is where
// they go, and keeping exposure fixed is what isolates that from a brightness
// change.
float ringEmission(float a) {
    vec4 p = scene.lightRingProfile;
    float w = 1.0 + p.z * cos(a - p.w);
    if (p.y > 0.0) {
        // Shortest angular distance to the gap centre.
        float d = abs(mod(a - p.x + PI, 2.0 * PI) - PI);
        w *= smoothstep(0.0, p.y, d);
    }
    return max(w, 0.0);
}

#endif // SURFACE_COMMON_GLSL
