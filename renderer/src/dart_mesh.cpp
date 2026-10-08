#include "dart_mesh.h"

#include "color.h"
#include "constants.h"

#include <algorithm>
#include <atomic>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dart {

namespace {

constexpr int kFloats = 12;
constexpr float kMM = (float)BU_PER_MM;

float uni(std::mt19937& rng, float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(rng);
}

int pickWeighted(std::mt19937& rng, const float* w, int n) {
    float total = 0.0f;
    for (int i = 0; i < n; ++i) total += w[i];
    float x = uni(rng, 0.0f, total);
    for (int i = 0; i < n; ++i) {
        x -= w[i];
        if (x <= 0.0f) return i;
    }
    return n - 1;
}

/// Triangular draw. Barrel weights cluster hard in 21-26g with thin tails
/// either side, so a uniform draw would make 18g and 30g as common as 23g.
float triangular(std::mt19937& rng, float lo, float mode, float hi) {
    const float u = uni(rng, 0.0f, 1.0f);
    const float c = (mode - lo) / std::max(hi - lo, 1e-6f);
    if (u < c) return lo + std::sqrt(u * (hi - lo) * (mode - lo));
    return hi - std::sqrt((1.0f - u) * (hi - lo) * (hi - mode));
}

float smoothstep01(float u) {
    u = std::min(std::max(u, 0.0f), 1.0f);
    return u * u * (3.0f - 2.0f * u);
}

// ---------------------------------------------------------------------------
// Barrel families
// ---------------------------------------------------------------------------

// (t, radius mm) from the FRONT of the barrel, where the point enters, to the
// REAR, where the shaft threads in.
//
// Every family carries an explicit shoulder at t=0.05 and t=0.95. A real
// barrel goes from the ~1.5mm nose to full diameter in 3-5mm; the nose taper
// is a chamfer, not a feature. Ramping over the first 10-30% of the length
// instead loses so much volume that calibrating against ten catalogue models
// implies a density of 20.6 g/cm3, above pure tungsten.
struct Family { glm::vec2 pts[8]; int n; float lenLo, lenHi; };

const Family kFamilies[kBarrelFamilyCount] = {
    // Parallel sided, the commonest tournament shape.
    {{{0.00f,1.50f},{0.05f,3.35f},{0.50f,3.40f},{0.95f,3.30f},{1.00f,2.80f}},
     5, 46.0f, 54.0f},
    // Slim and near-parallel. What makes a pencil a pencil is that it is LONG
    // for its weight, which the mass solve turns into a thin barrel by itself.
    {{{0.00f,1.45f},{0.05f,3.30f},{0.50f,3.35f},{0.95f,3.15f},{1.00f,2.70f}},
     5, 50.0f, 57.0f},
    // Widest forward of centre, tapering back.
    {{{0.00f,1.50f},{0.06f,3.05f},{0.30f,3.40f},{0.70f,2.95f},{0.95f,2.70f},
      {1.00f,2.30f}}, 6, 44.0f, 53.0f},
    // Bulb close to the point, then a long fall away.
    {{{0.00f,1.55f},{0.05f,2.95f},{0.22f,3.40f},{0.60f,2.85f},{0.95f,2.55f},
      {1.00f,2.20f}}, 6, 38.0f, 46.0f},
    // Two maxima with a waist between them, for a finger to sit in.
    {{{0.00f,1.50f},{0.05f,3.20f},{0.25f,3.40f},{0.50f,2.95f},{0.75f,3.40f},
      {0.95f,3.10f},{1.00f,2.65f}}, 7, 44.0f, 52.0f},
    // Mass toward the shaft; less common but sold.
    {{{0.00f,1.45f},{0.06f,2.65f},{0.35f,2.95f},{0.75f,3.40f},{0.95f,3.25f},
      {1.00f,2.75f}}, 6, 45.0f, 53.0f},
    // Long shallow cone, thick at the front.
    {{{0.00f,1.55f},{0.05f,3.40f},{0.55f,3.05f},{0.95f,2.75f},{1.00f,2.35f}},
     5, 42.0f, 51.0f},
};

// Material, density band, share, and the barrel-only weight band it sells in.
//
// Density is what sets diameter once weight is fixed, so this is the strongest
// single axis of variation between one player's set and another's. Brass is
// half tungsten's density, so a brass barrel of the same weight is ~35% fatter.
struct MaterialRow { float rhoLo, rhoHi, share, wLo, wMode, wHi; };
const MaterialRow kMaterials[3] = {
    {0.0f,  0.0f,  0.62f, 18.5f, 23.5f, 29.0f},   // tungsten: see kTungstenMix
    {8.35f, 8.65f, 0.33f, 17.0f, 22.0f, 27.0f},   // brass
    {8.60f, 8.90f, 0.05f, 16.0f, 20.5f, 25.0f},   // nickel silver
};

// Tungsten percentage -> alloy density, with each tier's share. The balance is
// nickel/iron/copper, which is why even 95% lands at 18.0 and not tungsten's
// own 19.3. 90% is the default tier and most of what is sold.
struct AlloyRow { float pct, rho, share; };
const AlloyRow kTungstenMix[5] = {
    {80.0f, 15.2f, 0.14f}, {85.0f, 16.1f, 0.12f}, {90.0f, 17.0f, 0.47f},
    {95.0f, 18.0f, 0.24f}, {97.0f, 18.4f, 0.03f},
};

// The two tapped holes, as (radius, depth) from each end. A barrel is not
// solid: the point presses into the nose and a 2BA shaft screws into the tail,
// and together they remove ~0.2cm3, which is 15% of a tungsten barrel's mass.
constexpr float kBoreFrontR = 1.25f, kBoreFrontD = 10.0f;
constexpr float kBoreRearR = 1.95f, kBoreRearD = 9.0f;

/// Catmull-Rom through (t, radius) control points, endpoints duplicated.
float catmullRadius(const glm::vec2* p, int n, float t) {
    if (n < 2) return n ? p[0].y : 0.0f;
    int i = 0;
    while (i < n - 2 && t > p[i + 1].x) ++i;
    const glm::vec2& p1 = p[i];
    const glm::vec2& p2 = p[i + 1];
    const glm::vec2& p0 = i > 0 ? p[i - 1] : p[0];
    const glm::vec2& p3 = (i + 2 < n) ? p[i + 2] : p[n - 1];
    const float u = (t - p1.x) / std::max(p2.x - p1.x, 1e-9f);
    const float u2 = u * u, u3 = u2 * u;
    const float r = 0.5f * (2.0f * p1.y
                            + (-p0.y + p2.y) * u
                            + (2.0f * p0.y - 5.0f * p1.y + 4.0f * p2.y - p3.y) * u2
                            + (-p0.y + 3.0f * p1.y - 3.0f * p2.y + p3.y) * u3);
    return std::max(r, 0.35f);
}

/// Scale the profile, leaving the NOSE alone.
///
/// Only the nose is pinned -- it meets a 2.35mm point shank whatever the
/// barrel weighs. The tail is not pinned the same way: a barrel ends in a
/// near-full-diameter flat, so it gets fatter with the rest of the barrel and
/// only has to stay clear of the 2BA thread inside it.
void applyGirth(const glm::vec2* in, int n, float girth, glm::vec2* out) {
    for (int i = 0; i < n; ++i)
        out[i] = glm::vec2(in[i].x, i == 0 ? in[i].y : in[i].y * girth);
}

} // namespace

float barrelMassG(const glm::vec2* ctrl, int nCtrl, float lengthMm,
                  float densityGcm3, float girth) {
    glm::vec2 g[8];
    applyGirth(ctrl, nCtrl, girth, g);
    constexpr int kN = 384;
    double sum = 0.0;
    for (int i = 0; i < kN; ++i) {
        const float t = ((float)i + 0.5f) / (float)kN;
        const float r = catmullRadius(g, nCtrl, t);
        const float z = t * lengthMm;
        float bore = 0.0f;
        if (z < kBoreFrontD) bore = std::max(bore, kBoreFrontR);
        if (z > lengthMm - kBoreRearD) bore = std::max(bore, kBoreRearR);
        const float b = std::min(bore, r);
        sum += std::max(r * r - b * b, 0.0f);
    }
    const double volMm3 = M_PI * (sum / kN) * lengthMm;
    return (float)(volMm3 * 1e-3 * densityGcm3);
}

namespace {

/// Girth that makes a profile weigh `weightG`.
///
/// GIRTH IS SOLVED, NOT DRAWN. Length and girth are not independent in a
/// catalogue: a barrel is sold by weight and density is fixed, so a short
/// barrel has to be fat and a long one thin to hit the same number. Drawing
/// them independently gives 4-11mm diameters against a real 6-8mm band.
///
/// The probes sit at 0.6/1.0/1.4 and not 0/1/2 because catmullRadius floors
/// its output at 0.35mm; at girth 0 every interior radius is under that floor
/// and the fitted parabola is wrong. The parabola is only a first guess in any
/// case -- min(bore, r) clamps wherever the barrel is thinner than its own
/// bore -- so secant steps follow it.
float solveGirth(const glm::vec2* ctrl, int n, float lengthMm, float rho,
                 float weightG) {
    const float gs[3] = {0.6f, 1.0f, 1.4f};
    float ms[3];
    for (int i = 0; i < 3; ++i)
        ms[i] = barrelMassG(ctrl, n, lengthMm, rho, gs[i]);
    // Quadratic through three equally spaced samples.
    const float a = (ms[2] - 2.0f * ms[1] + ms[0]) / (2.0f * 0.4f * 0.4f);
    const float b = (ms[2] - ms[0]) / (2.0f * 0.4f) - 2.0f * a * gs[1];
    const float c = ms[1] - a * gs[1] * gs[1] - b * gs[1] - weightG;
    float g = 1.0f;
    if (std::fabs(a) > 1e-9f) {
        const float disc = b * b - 4.0f * a * c;
        if (disc >= 0.0f) g = (-b + std::sqrt(disc)) / (2.0f * a);
    } else if (std::fabs(b) > 1e-9f) {
        g = -c / b;
    }
    g = std::min(std::max(g, 0.55f), 1.95f);

    float g1 = g, f1 = barrelMassG(ctrl, n, lengthMm, rho, g1) - weightG;
    float g0 = g * 0.97f;
    float f0 = barrelMassG(ctrl, n, lengthMm, rho, g0) - weightG;
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(f1 - f0) < 1e-9f) break;
        float g2 = g1 - f1 * (g1 - g0) / (f1 - f0);
        g2 = std::min(std::max(g2, 0.55f), 1.95f);
        g0 = g1; f0 = f1;
        g1 = g2; f1 = barrelMassG(ctrl, n, lengthMm, rho, g1) - weightG;
    }
    return g1;
}

// ---------------------------------------------------------------------------
// Grip
// ---------------------------------------------------------------------------

// Pitch, depth and cross-section per cut, with the share of barrels carrying
// each. Real ring grooves are cut 0.2-0.7mm deep; much shallower than that
// (under 3% of a 3.5mm radius) and the grip is indistinguishable from a smooth
// barrel.
struct CutRow { float pitchLo, pitchHi, depthLo, depthHi; GripCut kind;
                float share; bool ring; };
const CutRow kCuts[6] = {
    {0.55f, 0.95f, 0.16f, 0.30f, kCutRound,  0.32f, true},   // rings_fine
    {1.60f, 2.80f, 0.38f, 0.72f, kCutSquare, 0.20f, true},   // rings_bold
    {0.75f, 1.40f, 0.26f, 0.52f, kCutSaw,    0.16f, true},   // shark
    {0.33f, 0.55f, 0.13f, 0.26f, kCutRound,  0.14f, false},  // knurl
    {0.40f, 0.65f, 0.09f, 0.18f, kCutRound,  0.14f, true},   // micro
    {0.0f,  0.0f,  0.0f,  0.0f,  kCutNone,   0.04f, false},  // smooth
};

/// Share of barrels guaranteed to carry a ring cut somewhere. Drawing the
/// palette freely leaves only 61% with one, which undercounts badly: nearly
/// every dart sold has grooves somewhere, and a knurl-only or smooth barrel is
/// the exception, not a third of the catalogue. With this, 99.1% carry rings
/// (measured over 6000 draws).
constexpr float kPHasRings = 0.94f;

struct Zone { float a, b; };

int sampleZones(std::mt19937& rng, Zone* out) {
    const int layout = (int)(uni(rng, 0.0f, 6.0f));
    switch (layout < 0 ? 0 : (layout > 5 ? 5 : layout)) {
    case 0:  // full
        out[0] = {uni(rng, 0.04f, 0.10f), uni(rng, 0.90f, 0.97f)};
        return 1;
    case 1: { // centre
        const float h = uni(rng, 0.18f, 0.32f), c = uni(rng, 0.42f, 0.58f);
        out[0] = {std::max(c - h, 0.04f), std::min(c + h, 0.96f)};
        return 1;
    }
    case 2:  // two bands
        out[0] = {uni(rng, 0.10f, 0.20f), uni(rng, 0.34f, 0.44f)};
        out[1] = {uni(rng, 0.56f, 0.66f), uni(rng, 0.80f, 0.90f)};
        return 2;
    case 3: { // thirds -- abutting sections, where mixed cuts read clearest
        const float a = uni(rng, 0.05f, 0.10f);
        const float b = uni(rng, 0.32f, 0.40f);
        const float c = uni(rng, 0.60f, 0.68f);
        out[0] = {a, b}; out[1] = {b, c}; out[2] = {c, uni(rng, 0.90f, 0.96f)};
        return 3;
    }
    case 4: { // grouped bands
        const int n = 3 + (int)uni(rng, 0.0f, 3.0f);
        float t0 = uni(rng, 0.06f, 0.14f);
        int k = 0;
        for (int i = 0; i < n && k < 8; ++i) {
            const float w = uni(rng, 0.06f, 0.11f);
            if (t0 + w > 0.95f) break;
            out[k++] = {t0, t0 + w};
            t0 += w + uni(rng, 0.04f, 0.09f);
        }
        return k;
    }
    default: // front only
        out[0] = {uni(rng, 0.06f, 0.14f), uni(rng, 0.40f, 0.55f)};
        return 1;
    }
}

void sampleGrip(std::mt19937& rng, BarrelParams& bp) {
    Zone zones[8];
    int nZones = sampleZones(rng, zones);
    if (nZones <= 0) { bp.nZones = 0; return; }

    // How many distinct patterns this barrel carries.
    const float kindW[3] = {0.52f, 0.36f, 0.12f};
    int nKinds = pickWeighted(rng, kindW, 3) + 1;

    // Subdivide when there are not enough zones to carry them. Clamping
    // instead would drop the extra patterns, and since three of the six
    // layouts yield a single zone, mixed cuts would vanish on half of all
    // barrels (the 52/36/12 split comes out as 80/18/2). A continuous gripped
    // region whose pattern changes partway along is normal, so the region
    // splits rather than the pattern disappearing.
    while (nZones < nKinds && nZones < 8) {
        int widest = 0;
        for (int i = 1; i < nZones; ++i)
            if (zones[i].b - zones[i].a > zones[widest].b - zones[widest].a)
                widest = i;
        const float a = zones[widest].a, b = zones[widest].b;
        if (b - a < 0.06f) break;
        const float cut = a + (b - a) * uni(rng, 0.38f, 0.62f);
        for (int i = nZones; i > widest; --i) zones[i] = zones[i - 1];
        zones[widest] = {a, cut};
        zones[widest + 1] = {cut, b};
        ++nZones;
    }
    nKinds = std::min(nKinds, nZones);

    float shares[6];
    for (int i = 0; i < 6; ++i) shares[i] = kCuts[i].share;

    int palette[3];
    bool anyRing = false;
    for (int i = 0; i < nKinds; ++i) {
        palette[i] = pickWeighted(rng, shares, 6);
        anyRing |= kCuts[palette[i]].ring;
    }
    // Guarantee grooves somewhere. Replacing an existing entry rather than
    // appending keeps the 1/2/3-pattern split intact.
    if (!anyRing && uni(rng, 0.0f, 1.0f) < kPHasRings) {
        const float ringW[3] = {0.52f, 0.30f, 0.18f};   // fine / bold / micro
        const int map[3] = {0, 1, 4};
        palette[(int)uni(rng, 0.0f, (float)nKinds)] =
            map[pickWeighted(rng, ringW, 3)];
    }

    // Assign in contiguous RUNS: a barrel with grouped rings does not
    // alternate knurl and shark ring by ring, it carries one pattern for a
    // stretch and then changes.
    bp.nZones = nZones;
    for (int i = 0; i < nZones; ++i) {
        const int k = std::min(nKinds - 1,
                               (int)((float)i * nKinds / (float)nZones));
        const CutRow& c = kCuts[palette[k]];
        GripZone& gz = bp.zones[i];
        gz.a = zones[i].a;
        gz.b = zones[i].b;
        gz.cut = c.kind;
        gz.pitchMm = c.pitchHi > 0.0f ? uni(rng, c.pitchLo, c.pitchHi) : 0.0f;
        gz.depthMm = c.depthHi > 0.0f ? uni(rng, c.depthLo, c.depthHi) : 0.0f;
    }
}

/// Groove depth fraction in [0,1] within one period. 0 = crest, 1 = floor.
/// Phase runs from the barrel's FRONT toward its rear.
float cutProfile(float phase, GripCut kind) {
    switch (kind) {
    case kCutRound:
        return 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * phase);
    case kCutSquare:
        // A parting tool leaves a floor and slightly eased walls, not a knife
        // edge -- a true square wave reads as aliasing rather than as metal.
        return std::min(std::max((std::fabs(phase - 0.5f) - 0.22f) / -0.16f,
                                 0.0f), 1.0f);
    case kCutSaw:
        // Steep wall on the REAR face of each crest, gentle ramp in front of
        // it. The ridge has to catch a finger sliding toward the flight, which
        // is the direction a dart slips during the throw.
        return phase < 0.25f ? phase / 0.25f : (1.0f - phase) / 0.75f;
    default:
        return 0.0f;
    }
}

float barrelRadiusAt(const BarrelParams& bp, float t) {
    float r = catmullRadius(bp.ctrl, bp.nCtrl, t);
    for (int i = 0; i < bp.nZones; ++i) {
        const GripZone& z = bp.zones[i];
        if (z.depthMm <= 0.0f || z.pitchMm <= 0.0f) continue;
        if (t < z.a || t > z.b) continue;
        const float span = std::max(z.b - z.a, 1e-6f);
        // Phase from each zone's own start, so a change of pattern does not
        // inherit a partial period from the zone before it.
        float phase = std::fmod((t - z.a) * bp.lengthMm / z.pitchMm, 1.0f);
        if (phase < 0.0f) phase += 1.0f;
        const float u = (t - z.a) / span;
        const float ease = std::min(std::min(u, 1.0f - u) / 0.06f, 1.0f);
        r -= z.depthMm * cutProfile(phase, z.cut) * std::max(ease, 0.0f);
    }
    return std::max(r, 0.35f);
}

// ---------------------------------------------------------------------------
// Point
// ---------------------------------------------------------------------------

constexpr float kPointEngagedMm = 8.0f;

float pointRadiusAt(const PointParams& pp, float z) {
    float r;
    if (z <= pp.shankMm) {
        const float u = std::min(std::max(z / std::max(pp.shankMm, 1e-6f),
                                          0.0f), 1.0f);
        r = pp.rShank + (pp.rNeck - pp.rShank) * u;
    } else {
        const float u = std::min(std::max((z - pp.shankMm)
                                          / std::max(pp.taperMm, 1e-6f),
                                          0.0f), 1.0f);
        r = pp.rTip + (pp.rNeck - pp.rTip) * std::pow(1.0f - u, pp.exponent);
    }
    if (pp.cut != kCutNone && pp.gripDepthMm > 0.0f && pp.gripPitchMm > 0.0f
        && z >= pp.gripA && z <= pp.gripB) {
        float phase = std::fmod((z - pp.gripA) / pp.gripPitchMm, 1.0f);
        if (phase < 0.0f) phase += 1.0f;
        const float u = (z - pp.gripA) / std::max(pp.gripB - pp.gripA, 1e-6f);
        const float ease = std::min(std::min(u, 1.0f - u) / 0.08f, 1.0f);
        r -= pp.gripDepthMm * cutProfile(phase, pp.cut) * std::max(ease, 0.0f);
    }
    return std::max(r, 0.012f);
}

// ---------------------------------------------------------------------------
// Tail
// ---------------------------------------------------------------------------

/// Fraction of the flight over which the shaft envelope fades out behind the
/// junction, the core and blends run to zero, and the vane thins.
constexpr float kEnvFade = 0.12f;
constexpr float kTipClose = 0.10f;
constexpr float kTTaperFrac = 0.12f;
constexpr float kValleyTaperEnd = 0.30f;
constexpr float kLeadK = 0.55f;
constexpr int kOutlineN = 256;

float smax(float a, float b, float k) {
    const float h = std::min(std::max(0.5f + 0.5f * (a - b) / k, 0.0f), 1.0f);
    return b + (a - b) * h + k * h * (1.0f - h);
}

struct TailOutline { float w[kOutlineN]; };

/// The kite outline: four control points with straight edges between them,
/// then a Gaussian blur in u.
///
/// The blur is what rounds the corners, and it has exactly the property wanted
/// here -- a linear function is invariant under a symmetric kernel, so the
/// straight runs stay straight and only the corners round. A spline through
/// the same points curves the edges as well and cannot be dialled down.
/// Endpoints are edge-padded so the root width and the zero at the tip both
/// survive it.
void buildOutline(const TailParams& tp, TailOutline& out) {
    float lin[kOutlineN];
    float peak = 1e-9f;
    for (int i = 0; i < kOutlineN; ++i) {
        const float u = (float)i / (float)(kOutlineN - 1);
        int s = 0;
        while (s < 2 && u > tp.kite[s + 1].x) ++s;
        const glm::vec2 a = tp.kite[s], b = tp.kite[s + 1];
        const float f = (u - a.x) / std::max(b.x - a.x, 1e-9f);
        lin[i] = a.y + (b.y - a.y) * std::min(std::max(f, 0.0f), 1.0f);
        peak = std::max(peak, lin[i]);
    }
    const float du = 1.0f / (float)(kOutlineN - 1);
    const float sigma = tp.smooth;
    if (sigma <= 2.0f * du) {
        for (int i = 0; i < kOutlineN; ++i) out.w[i] = lin[i] / peak;
        out.w[kOutlineN - 1] = 0.0f;
        return;
    }
    const int rad = (int)std::ceil(3.0f * sigma / du);
    float newPeak = 1e-9f;
    for (int i = 0; i < kOutlineN; ++i) {
        float acc = 0.0f, wsum = 0.0f;
        for (int j = -rad; j <= rad; ++j) {
            const float x = (float)j * du;
            const float k = std::exp(-0.5f * (x / sigma) * (x / sigma));
            const int idx = std::min(std::max(i + j, 0), kOutlineN - 1);
            acc += k * lin[idx];
            wsum += k;
        }
        out.w[i] = acc / std::max(wsum, 1e-9f);
        newPeak = std::max(newPeak, out.w[i]);
    }
    for (int i = 0; i < kOutlineN; ++i) out.w[i] = std::max(out.w[i], 0.0f) / newPeak;
    out.w[kOutlineN - 1] = 0.0f;
}

float aerofoilR(const TailParams& tp, const TailOutline& o, float z) {
    const float u = std::min(std::max((z - tp.zFlightMm)
                                      / std::max(tp.flightMm, 1e-6f),
                                      0.0f), 1.0f);
    const float f = u * (float)(kOutlineN - 1);
    const int i = std::min((int)f, kOutlineN - 2);
    const float frac = f - (float)i;
    return (o.w[i] + (o.w[i + 1] - o.w[i]) * frac) * tp.spanMm * 0.5f;
}

/// Outer radius of the shaft, which the VANES carry once the ribs start, so
/// from outside the stem still reads as a shaft of the right diameter while
/// its cross-section is a cross with concave valleys.
float shaftEnvelope(const TailParams& tp, float z) {
    const float u = std::min(std::max(z / std::max(tp.stemMm, 1e-6f),
                                      0.0f), 1.0f);
    return tp.r0 + (tp.r1 - tp.r0) * u;
}

float vaneOuter(const TailParams& tp, const TailOutline& o, float z) {
    const float zr = tp.ribFrac * tp.stemMm;
    if (z < zr) return 0.0f;
    const float fadeSpan = std::max(kEnvFade * tp.flightMm, 1e-6f);
    const float fade = smoothstep01((tp.zFlightMm + fadeSpan - z) / fadeSpan);
    const float env = shaftEnvelope(tp, z) * (z <= tp.zFlightMm ? 1.0f : fade);
    // The leading-edge blend has to close too: a soft maximum of two zeros is
    // k/4, so with a fixed blend the tip would stay at 0.1375mm however far
    // the outline converges.
    const float close = std::min(std::max((tp.totalMm - z)
                                          / std::max(kTipClose * tp.flightMm,
                                                     1e-6f), 0.0f), 1.0f);
    return smax(env, aerofoilR(tp, o, z), std::max(kLeadK * close, 1e-6f));
}

/// Round core: full diameter at the barrel, gone soon after the ribs. It runs
/// out to ZERO at the tip -- ending on a finite spine radius would leave the
/// last ring open and the mesh finishing on a flat disc instead of closing.
float coreRadius(const TailParams& tp, float z) {
    const float zr = tp.ribFrac * tp.stemMm;
    const float zs[5] = {0.0f, zr, zr + 0.45f * (tp.stemMm - zr),
                         tp.totalMm - kTipClose * tp.flightMm, tp.totalMm};
    const float rs[5] = {tp.r0, tp.r0 * 0.99f, tp.spineR, tp.spineR * 0.88f,
                         0.0f};
    if (z <= zs[0]) return rs[0];
    for (int i = 0; i < 4; ++i) {
        if (z <= zs[i + 1]) {
            const float f = (z - zs[i]) / std::max(zs[i + 1] - zs[i], 1e-9f);
            return rs[i] + (rs[i + 1] - rs[i]) * f;
        }
    }
    return rs[4];
}

/// Arm thickness: stout where the ribs leave the stem, thin at the vane. The
/// taper happens AT the junction, not along the shaft -- the shaft rib keeps
/// its section right up to the aerofoil and the drop is a local feature.
float vaneThickness(const TailParams& tp, float z) {
    const float u = smoothstep01((z - tp.zFlightMm)
                                 / std::max(kTTaperFrac * tp.flightMm, 1e-6f));
    return tp.tRootMm + (tp.vaneTMm - tp.tRootMm) * u;
}

/// Concave corner fillet between adjacent vanes. Tapers from the RIB start,
/// not from the aerofoil: holding the shaft value to the junction puts the
/// fillet's tangent point (~2.05mm) almost where the aerofoil leaves the stem
/// (~2.14mm), so the whole vane root becomes fillet -- a blob.
float valleyFillet(const TailParams& tp, float z) {
    const float zr = tp.ribFrac * tp.stemMm;
    const float end = tp.zFlightMm + kValleyTaperEnd * tp.flightMm;
    const float u = std::min(std::max((z - zr) / std::max(end - zr, 1e-6f),
                                      0.0f), 1.0f);
    return tp.valleyF0 + (tp.valleyF1 - tp.valleyF0) * u;
}

/// Four 90-degree plates with concave corner fillets.
///
/// Working in the quadrant between the vane at 0 and the vane at 90, with
/// plate faces at y = t/2 and x = t/2, the fillet is an arc of radius f centred
/// at C = (t/2 + f, t/2 + f). A ray at angle `delta` from the vane axis exits
/// on that arc whenever the plate-face hit would land inside the tangent point,
/// i.e. when (t/2)/tan(delta) <= t/2 + f; otherwise it exits on the flat face
/// at (t/2)/sin(delta). Both the plate and the valley fall out of this; neither
/// is modelled separately.
float vaneSection(float delta, float t, float f, float rOut) {
    const float half = 0.5f * t;
    const float sd = std::max(std::sin(delta), 1e-9f);
    const float td = std::max(std::tan(delta), 1e-9f);
    const float rPlate = half / sd;
    const float cx = half + f;
    const float cu = cx * (std::cos(delta) + std::sin(delta));
    const float disc = std::max(cu * cu - 2.0f * cx * cx + f * f, 0.0f);
    const float rArc = cu - std::sqrt(disc);
    const float r = (half / td) <= cx ? rArc : rPlate;
    return std::min(r, rOut);
}

float tailRadius(const TailParams& tp, const TailOutline& o, float z,
                 float theta) {
    const float rc = coreRadius(tp, z);
    const float ro = vaneOuter(tp, o, z);
    // Angle to the nearest vane plane. The planes are two-sided, so the period
    // is 90 degrees and the distance folds at 45.
    const float base = glm::radians(tp.rollDeg);
    float d = std::fmod(theta - base + 0.25f * (float)M_PI, 0.5f * (float)M_PI);
    if (d < 0.0f) d += 0.5f * (float)M_PI;
    const float delta = std::fabs(d - 0.25f * (float)M_PI);

    float blade = 0.0f;
    if (ro > rc)
        blade = vaneSection(delta, vaneThickness(tp, z), valleyFillet(tp, z), ro);
    const float close = std::min(std::max((tp.totalMm - z)
                                          / std::max(kTipClose * tp.flightMm,
                                                     1e-6f), 0.0f), 1.0f);
    return smax(rc, blade, std::max(tp.filletMm * close, 1e-6f));
}

} // namespace

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

namespace {

void sampleBarrel(std::mt19937& rng, BarrelParams& bp) {
    bp.family = (BarrelFamily)((int)uni(rng, 0.0f, (float)kBarrelFamilyCount)
                               % kBarrelFamilyCount);
    const Family& fam = kFamilies[bp.family];
    bp.lengthMm = uni(rng, fam.lenLo, fam.lenHi);

    // Jitter, in the two ways a catalogue actually varies: small per-point
    // noise so two barrels of the same family and weight are not identical,
    // and a shift of where the mass sits.
    std::normal_distribution<float> jitter(0.0f, 0.055f);
    glm::vec2 pts[8];
    for (int i = 0; i < fam.n; ++i)
        pts[i] = glm::vec2(fam.pts[i].x, fam.pts[i].y * (1.0f + jitter(rng)));
    pts[0].y = fam.pts[0].y * uni(rng, 0.94f, 1.08f);
    pts[fam.n - 1].y = fam.pts[fam.n - 1].y * uni(rng, 0.94f, 1.08f);

    // The shoulders sit 0.05 from each end, so a larger shift could slide one
    // onto the end point and collapse the transition.
    const float tshift = uni(rng, -0.035f, 0.035f);
    for (int i = 1; i < fam.n - 1; ++i)
        pts[i].x = std::min(std::max(pts[i].x + tshift, 0.035f), 0.965f);
    std::sort(pts, pts + fam.n,
              [](const glm::vec2& a, const glm::vec2& b) { return a.x < b.x; });

    const float matW[3] = {kMaterials[0].share, kMaterials[1].share,
                           kMaterials[2].share};
    bp.material = (BarrelMaterial)pickWeighted(rng, matW, 3);
    const MaterialRow& m = kMaterials[bp.material];
    if (bp.material == kTungsten) {
        float w[5];
        for (int i = 0; i < 5; ++i) w[i] = kTungstenMix[i].share;
        const AlloyRow& a = kTungstenMix[pickWeighted(rng, w, 5)];
        bp.alloyPct = a.pct;
        bp.densityGcm3 = a.rho * uni(rng, 0.98f, 1.02f);
    } else {
        bp.alloyPct = 0.0f;
        bp.densityGcm3 = uni(rng, m.rhoLo, m.rhoHi);
    }
    bp.weightG = triangular(rng, m.wLo, m.wMode, m.wHi);

    bp.girth = solveGirth(pts, fam.n, bp.lengthMm, bp.densityGcm3, bp.weightG);
    applyGirth(pts, fam.n, bp.girth, bp.ctrl);
    bp.nCtrl = fam.n;

    sampleGrip(rng, bp);
}

void samplePoint(std::mt19937& rng, PointParams& pp) {
    // 30 / 35 / 40-42mm are the sizes sold. 50mm exists but is a specialist
    // item, and would give a point longer than half the barrel behind it.
    const float lens[3] = {30.0f, 35.0f, 41.0f};
    const float lw[3] = {0.30f, 0.42f, 0.28f};
    pp.quotedMm = lens[pickWeighted(rng, lw, 3)] + uni(rng, -1.0f, 1.0f);
    pp.exposedMm = pp.quotedMm - kPointEngagedMm;

    // Cone or bullet. A concave needle profile does not look like a real
    // point, so the exponent band stops at 1.15.
    const float profW[2] = {0.62f, 0.38f};
    pp.exponent = pickWeighted(rng, profW, 2) == 0 ? uni(rng, 0.90f, 1.15f)
                                                   : uni(rng, 0.52f, 0.85f);

    // The taper stays 13-22mm whatever the size; extra length goes into the
    // shank. The upper end is lowered so a 30mm point is not left with no
    // shank at all -- lowered rather than clamped, which would stack every
    // over-long draw onto one exact taper length.
    pp.taperMm = uni(rng, 13.0f, std::min(22.0f, pp.exposedMm * 0.78f));
    pp.shankMm = pp.exposedMm - pp.taperMm;

    pp.rShank = uni(rng, 1.08f, 1.38f);
    pp.rNeck = pp.rShank * uni(rng, 0.86f, 1.00f);

    // Even a factory point is not a mathematical point, and most points in
    // play are blunter still: they get dressed with a stone and hit wires.
    const float tipW[3] = {0.32f, 0.43f, 0.25f};
    switch (pickWeighted(rng, tipW, 3)) {
    case 0: pp.rTip = uni(rng, 0.10f, 0.18f); break;
    case 1: pp.rTip = uni(rng, 0.18f, 0.32f); break;
    default: pp.rTip = uni(rng, 0.32f, 0.55f); break;
    }

    const float gripW[3] = {0.40f, 0.36f, 0.24f};   // smooth / rings / knurl
    const int g = pickWeighted(rng, gripW, 3);
    if (g == 0) {
        pp.cut = kCutNone;
    } else {
        pp.cut = kCutRound;
        pp.gripPitchMm = g == 1 ? uni(rng, 0.55f, 1.10f) : uni(rng, 0.30f, 0.50f);
        pp.gripDepthMm = g == 1 ? uni(rng, 0.055f, 0.130f)
                                : uni(rng, 0.035f, 0.075f);
        pp.gripA = uni(rng, 0.05f, 0.30f) * pp.shankMm;
        pp.gripB = pp.gripA + uni(rng, 0.45f, 0.92f) * (pp.shankMm - pp.gripA);
    }

    const float finW[5] = {0.46f, 0.21f, 0.15f, 0.11f, 0.07f};
    pp.finish = (PointFinish)pickWeighted(rng, finW, 5);
}

void sampleTail(std::mt19937& rng, TailParams& tp) {
    // Discrete stem sizes, and ONLY the stem changes: the moulded flight is
    // the same size in every length of a given model.
    const float stems[4] = {16.0f, 21.5f, 27.5f, 34.5f};
    const float sw[4] = {0.09f, 0.26f, 0.36f, 0.29f};
    tp.stemMm = stems[pickWeighted(rng, sw, 4)] + uni(rng, -0.8f, 0.8f);

    tp.flightMm = uni(rng, 34.0f, 41.0f);
    tp.spanMm = uni(rng, 24.5f, 30.5f);
    tp.zFlightMm = tp.stemMm;              // no slot on a one-piece
    tp.totalMm = tp.stemMm + tp.flightMm;

    // The last free point sets how the flight ends. Most moulded flights stay
    // wide almost to the back and fall to the axis over a short trailing edge,
    // so it is drawn toward the rear and toward full width (sqrt of a uniform
    // leans to the top of each range). A long taper to a point -- the kite and
    // slim shapes -- remains, as the minority it is in use.
    const float u1 = uni(rng, 0.16f, 0.40f);
    const float u2lo = u1 + 0.16f;
    const float u2 = u2lo + (0.95f - u2lo) * std::sqrt(uni(rng, 0.0f, 1.0f));
    tp.kite[0] = glm::vec2(0.0f, uni(rng, 0.04f, 0.12f));
    tp.kite[1] = glm::vec2(u1, uni(rng, 0.45f, 0.95f));
    tp.kite[2] = glm::vec2(u2, 0.58f + 0.42f * std::sqrt(uni(rng, 0.0f, 1.0f)));
    tp.kite[3] = glm::vec2(1.0f, 0.0f);    // the tip: ONE point, on the axis
    tp.smooth = uni(rng, 0.008f, 0.035f);

    tp.r1 = uni(rng, 1.95f, 2.45f);
    tp.spineR = uni(rng, 0.28f, 0.55f);
    tp.ribFrac = uni(rng, 0.12f, 0.26f);
    tp.tRootMm = uni(rng, 1.05f, 1.75f);
    tp.vaneTMm = uni(rng, 0.056f, 0.100f);
    tp.valleyF0 = uni(rng, 0.80f, 1.30f);
    tp.valleyF1 = uni(rng, 0.22f, 0.42f);
    tp.filletMm = uni(rng, 0.30f, 0.65f);
    tp.rollDeg = uni(rng, 0.0f, 90.0f);

    // Moulded flight colours, sRGB bytes converted to linear here rather than
    // in a shader. Using sRGB values directly as linear albedo lifts the dark
    // end by up to 10x.
    const glm::vec3 srgb[8] = {
        {28, 28, 30}, {226, 226, 230}, {198, 32, 38}, {32, 72, 188},
        {240, 200, 44}, {24, 132, 78}, {128, 40, 160}, {250, 128, 24}};
    const glm::vec3 c = srgb[(int)uni(rng, 0.0f, 8.0f) % 8] / 255.0f;
    tp.colorLinear = srgbToLinear(c);
    tp.flightRoughness = uni(rng, 0.24f, 0.58f);
    // Most moulded flights are solid; a minority are translucent, and those
    // cast a coloured shadow rather than merely a weaker one.
    tp.flightAlpha = uni(rng, 0.0f, 1.0f) < 0.26f ? uni(rng, 0.45f, 0.88f)
                                                  : 1.0f;
}

} // namespace

DartParams sampleDart(std::mt19937& rng) {
    DartParams p;
    sampleBarrel(rng, p.barrel);
    samplePoint(rng, p.point);
    sampleTail(rng, p.tail);
    // The shaft collar is made to match the barrel it screws into, so the tail
    // takes its front radius from the barrel's rear rather than drawing one.
    p.tail.r0 = std::max(p.barrel.ctrl[p.barrel.nCtrl - 1].y, p.tail.r1 + 0.15f);
    return p;
}

// ---------------------------------------------------------------------------
// Mesh construction
// ---------------------------------------------------------------------------

namespace {

/// Sweep a radius profile r(z) around +Z. `z` and `r` are in millimetres;
/// output is board units.
///
/// Normals are analytic from the profile slope -- for a surface of revolution
/// the outward normal is (cos a, sin a, -dr/dz) normalised. Averaging face
/// normals would round the grip crests, which are the whole point of cutting
/// them into the geometry rather than into a roughness map.
void lathe(const std::vector<float>& z, const std::vector<float>& r,
           int segments, bool capFront, bool capBack, DartPartMesh& out) {
    const int nz = (int)z.size();
    if (nz < 2 || segments < 3) return;
    const uint32_t base = (uint32_t)(out.vertices.size() / kFloats);
    const float zLen = std::max(z.back() - z.front(), 1e-6f);

    for (int i = 0; i < nz; ++i) {
        // One-sided at the ends, central inside.
        const int i0 = std::max(i - 1, 0), i1 = std::min(i + 1, nz - 1);
        const float drdz = (r[i1] - r[i0]) / std::max(z[i1] - z[i0], 1e-9f);
        const float nz_ = -drdz;
        const float inv = 1.0f / std::sqrt(1.0f + drdz * drdz);
        for (int s = 0; s < segments; ++s) {
            const float a = 2.0f * (float)M_PI * (float)s / (float)segments;
            const float ca = std::cos(a), sa = std::sin(a);
            out.vertices.insert(out.vertices.end(), {
                r[i] * ca * kMM, r[i] * sa * kMM, z[i] * kMM,
                ca * inv, sa * inv, nz_ * inv,
                (float)s / (float)segments, (z[i] - z.front()) / zLen,
                -sa, ca, 0.0f, 1.0f});
        }
    }
    for (int i = 0; i + 1 < nz; ++i) {
        for (int s = 0; s < segments; ++s) {
            const uint32_t s1 = (uint32_t)((s + 1) % segments);
            const uint32_t a = base + (uint32_t)(i * segments + s);
            const uint32_t b = base + (uint32_t)(i * segments) + s1;
            const uint32_t c = base + (uint32_t)((i + 1) * segments + s);
            const uint32_t d = base + (uint32_t)((i + 1) * segments) + s1;
            out.indices.insert(out.indices.end(), {a, c, d, a, d, b});
        }
    }
    // Caps keep the shell closed. An open shell gives ray queries a way into
    // the interior, and an ambient or shadow ray that enters one end and exits
    // the missing other reports occlusion that is not there.
    auto cap = [&](int row, float sign) {
        const uint32_t centre = (uint32_t)(out.vertices.size() / kFloats);
        out.vertices.insert(out.vertices.end(), {
            0.0f, 0.0f, z[row] * kMM, 0.0f, 0.0f, sign,
            0.5f, sign > 0.0f ? 1.0f : 0.0f, 1.0f, 0.0f, 0.0f, 1.0f});
        const uint32_t ring = base + (uint32_t)(row * segments);
        for (int s = 0; s < segments; ++s) {
            const uint32_t a = ring + (uint32_t)s;
            const uint32_t b = ring + (uint32_t)((s + 1) % segments);
            if (sign > 0.0f) out.indices.insert(out.indices.end(), {a, b, centre});
            else             out.indices.insert(out.indices.end(), {b, a, centre});
        }
    };
    if (capFront) cap(0, -1.0f);
    if (capBack) cap(nz - 1, 1.0f);
}

} // namespace

void buildDart(const DartParams& p, const DartTessellation& tess,
               DartBuild& out) {
    out = DartBuild{};

    // ---- Point ------------------------------------------------------------
    //
    // z runs from the tip at 0 back to the barrel, so the profile is evaluated
    // reversed: pointRadiusAt measures from the barrel.
    {
        const int n = std::max(tess.pointRings, 16);
        std::vector<float> z(n), r(n);
        for (int i = 0; i < n; ++i) {
            // Cluster toward the tip: that end has all the curvature, and a
            // uniform sweep cuts the point into a small flat facet.
            const float f = (float)i / (float)(n - 1);
            const float zFromBarrel = p.point.exposedMm * (1.0f - f * f);
            z[i] = p.point.exposedMm - zFromBarrel;
            r[i] = pointRadiusAt(p.point, zFromBarrel);
        }
        lathe(z, r, tess.pointSegments, false, true, out.point);
    }

    // ---- Barrel -----------------------------------------------------------
    const float barrelZ0 = p.point.exposedMm;
    {
        // Rings enough to resolve the FINEST cut on this barrel, not an
        // average: one knurled band at a 0.33mm pitch sets the tessellation
        // for the whole barrel, and budgeting for the mean is how a grip
        // pattern silently turns into noise.
        float finest = 1e9f;
        for (int i = 0; i < p.barrel.nZones; ++i)
            if (p.barrel.zones[i].pitchMm > 0.0f)
                finest = std::min(finest, p.barrel.zones[i].pitchMm);
        int n = tess.barrelMinRings;
        if (finest < 1e8f)
            n = std::max(n, (int)std::ceil(tess.gripSamplesPerPeriod
                                           * p.barrel.lengthMm / finest));
        n = std::min(n, 2048);

        std::vector<float> z(n), r(n);
        for (int i = 0; i < n; ++i) {
            const float t = (float)i / (float)(n - 1);
            z[i] = barrelZ0 + t * p.barrel.lengthMm;
            r[i] = barrelRadiusAt(p.barrel, t);
            out.barrelDiaMm = std::max(out.barrelDiaMm, 2.0f * r[i]);
        }
        lathe(z, r, tess.barrelSegments, false, false, out.barrel);
    }

    // ---- Tail -------------------------------------------------------------
    const float tailZ0 = barrelZ0 + p.barrel.lengthMm;
    {
        const TailParams& tp = p.tail;
        TailOutline outline;
        buildOutline(tp, outline);

        // Angular samples clustered toward each vane plane, and again toward
        // the 45-degree valley floor, which has its own curvature.
        std::vector<float> th;
        const int nh = std::max(tess.tailAngularPerHalfQuadrant, 6);
        std::vector<float> off;
        for (int i = 0; i < nh; ++i) {
            const float u = (float)i / (float)nh;
            off.push_back(glm::radians(45.0f) * u * u * u);
        }
        for (int i = 1; i <= 6; ++i) {
            const float w = (float)i / 6.0f;
            const float d = glm::radians(45.0f) - glm::radians(11.0f) * w * w;
            if (d > off.back()) off.push_back(d);
        }
        std::sort(off.begin(), off.end());
        const float base = glm::radians(tp.rollDeg);
        for (int q = 0; q < 4; ++q) {
            const float c = base + q * 0.5f * (float)M_PI;
            for (size_t i = off.size(); i-- > 1;) th.push_back(c - off[i]);
            for (size_t i = 0; i < off.size(); ++i) th.push_back(c + off[i]);
        }
        std::sort(th.begin(), th.end());
        th.erase(std::unique(th.begin(), th.end(),
                             [](float a, float b) { return std::fabs(a - b) < 1e-7f; }),
                 th.end());

        // Axial samples: graded through the flight, landing on the outline's
        // corners, and clustered into the last 2mm where the tip closes.
        std::vector<float> zs;
        const int nStem = std::max(tess.tailAxial / 3, 8);
        const int nFly = std::max(tess.tailAxial - nStem, 16);
        const float zr = tp.ribFrac * tp.stemMm;
        for (int i = 0; i < nStem; ++i)
            zs.push_back(tp.zFlightMm * (float)i / (float)(nStem - 1));
        for (int i = 0; i < nFly; ++i) {
            const float u = (float)i / (float)(nFly - 1);
            zs.push_back(tp.zFlightMm
                         + (tp.totalMm - tp.zFlightMm) * std::pow(u, 0.85f));
        }
        zs.push_back(zr - 1e-4f);
        zs.push_back(zr + 1e-4f);
        for (int k = 1; k <= 2; ++k) {
            const float cu = tp.kite[k].x;
            zs.push_back(tp.zFlightMm + cu * tp.flightMm - 1e-4f);
            zs.push_back(tp.zFlightMm + cu * tp.flightMm + 1e-4f);
        }
        for (int i = 0; i < 12; ++i) {
            const float f = (float)i / 11.0f;
            zs.push_back(tp.totalMm - 2.0f * f * f);
        }
        std::sort(zs.begin(), zs.end());
        zs.erase(std::remove_if(zs.begin(), zs.end(),
                                [&](float v) { return v < 0.0f || v > tp.totalMm; }),
                 zs.end());
        zs.erase(std::unique(zs.begin(), zs.end(),
                             [](float a, float b) { return std::fabs(a - b) < 1e-5f; }),
                 zs.end());

        const int nz = (int)zs.size(), na = (int)th.size();
        std::vector<float> R((size_t)nz * na);
        for (int i = 0; i < nz; ++i)
            for (int j = 0; j < na; ++j)
                R[(size_t)i * na + j] = tailRadius(tp, outline, zs[i], th[j]);

        // Normals analytically from the radius field. For a surface r(z,th),
        // n = dP/dth x dP/dz. Averaging face normals would smear the two faces
        // of a 0.08mm plate together, which is precisely the feature the
        // angular grading exists to resolve.
        DartPartMesh& m = out.tail;
        const uint32_t vbase = (uint32_t)(m.vertices.size() / kFloats);
        for (int i = 0; i < nz; ++i) {
            for (int j = 0; j < na; ++j) {
                const float r = R[(size_t)i * na + j];
                const float a = th[j];
                const int i0 = std::max(i - 1, 0), i1 = std::min(i + 1, nz - 1);
                const int j0 = (j - 1 + na) % na, j1 = (j + 1) % na;
                const float drdz = (R[(size_t)i1 * na + j] - R[(size_t)i0 * na + j])
                                 / std::max(zs[i1] - zs[i0], 1e-9f);
                float dth = th[j1] - th[j0];
                if (dth < 0.0f) dth += 2.0f * (float)M_PI;
                const float drdth = (R[(size_t)i * na + j1] - R[(size_t)i * na + j0])
                                  / std::max(dth, 1e-9f);
                const float ca = std::cos(a), sa = std::sin(a);
                const glm::vec3 dPdth(drdth * ca - r * sa, drdth * sa + r * ca, 0.0f);
                const glm::vec3 dPdz(drdz * ca, drdz * sa, 1.0f);
                glm::vec3 n = glm::cross(dPdth, dPdz);
                const float len = glm::length(n);
                n = len > 1e-12f ? n / len : glm::vec3(ca, sa, 0.0f);
                glm::vec3 tg = glm::normalize(glm::vec3(-sa, ca, 0.0f));
                m.vertices.insert(m.vertices.end(), {
                    r * ca * kMM, r * sa * kMM, (tailZ0 + zs[i]) * kMM,
                    n.x, n.y, n.z,
                    a / (2.0f * (float)M_PI), zs[i] / std::max(tp.totalMm, 1e-6f),
                    tg.x, tg.y, tg.z, 1.0f});
            }
        }
        for (int i = 0; i + 1 < nz; ++i) {
            for (int j = 0; j < na; ++j) {
                const uint32_t j1 = (uint32_t)((j + 1) % na);
                const uint32_t a = vbase + (uint32_t)(i * na + j);
                const uint32_t b = vbase + (uint32_t)(i * na) + j1;
                const uint32_t c = vbase + (uint32_t)((i + 1) * na + j);
                const uint32_t d = vbase + (uint32_t)((i + 1) * na) + j1;
                m.indices.insert(m.indices.end(), {a, c, d, a, d, b});
            }
        }
        // Front cap only. The tail end closes on the axis by construction, so
        // a cap there would be zero-area.
        const uint32_t centre = (uint32_t)(m.vertices.size() / kFloats);
        m.vertices.insert(m.vertices.end(), {
            0.0f, 0.0f, tailZ0 * kMM, 0.0f, 0.0f, -1.0f,
            0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f});
        for (int j = 0; j < na; ++j) {
            const uint32_t a = vbase + (uint32_t)j;
            const uint32_t b = vbase + (uint32_t)((j + 1) % na);
            m.indices.insert(m.indices.end(), {b, a, centre});
        }

        out.finish.flightColor = tp.colorLinear;
    }

    // ---- Derived ----------------------------------------------------------
    out.tipLocal = glm::vec3(0.0f);
    out.tailLocal = glm::vec3(0.0f, 0.0f, (tailZ0 + p.tail.totalMm) * kMM);

    out.barrelLengthMm = p.barrel.lengthMm;
    out.totalLengthMm = tailZ0 + p.tail.totalMm;

    // Metal F0 at textbook values. These rely on the shaders gating ambient
    // diffuse by (1 - metallic); without that gate every barrel renders white.
    switch (p.barrel.material) {
    case kTungsten:
        out.finish.metalColor = glm::vec3(0.45f, 0.44f, 0.42f);
        // A budget 80% alloy is not just fatter, it is duller: the balance
        // metal is what takes the polish badly.
        out.finish.metalRoughness =
            0.42f + 0.30f * (1.0f - p.barrel.alloyPct / 97.0f);
        break;
    case kBrass:
        out.finish.metalColor = glm::vec3(0.63f, 0.50f, 0.24f);
        out.finish.metalRoughness = 0.38f;
        break;
    default:
        out.finish.metalColor = glm::vec3(0.66f, 0.66f, 0.68f);
        out.finish.metalRoughness = 0.34f;
        break;
    }
    switch (p.point.finish) {
    case kSteel:      out.finish.pointColor = glm::vec3(0.56f, 0.57f, 0.58f);
                      out.finish.pointRoughness = 0.36f; break;
    case kBlackCoat:  out.finish.pointColor = glm::vec3(0.08f, 0.08f, 0.09f);
                      out.finish.pointRoughness = 0.52f; break;
    case kGoldCoat:   out.finish.pointColor = glm::vec3(0.72f, 0.56f, 0.24f);
                      out.finish.pointRoughness = 0.30f; break;
    case kSilverCoat: out.finish.pointColor = glm::vec3(0.70f, 0.71f, 0.72f);
                      out.finish.pointRoughness = 0.26f; break;
    default:          out.finish.pointColor = glm::vec3(0.31f, 0.32f, 0.34f);
                      out.finish.pointRoughness = 0.48f; break;
    }
    out.finish.flightColor = p.tail.colorLinear;
    out.finish.flightRoughness = p.tail.flightRoughness;
    out.finish.flightAlpha = p.tail.flightAlpha;
}

void extractGeometry(const DartBuild& b, DartGeometry& g) {
    g.tipLocal = b.tipLocal;
    g.tailLocal = b.tailLocal;
    g.finish = b.finish;
    // Starts at 1 so 0 stays free to mean "never extracted".
    static std::atomic<uint64_t> nextGeneration{1};
    g.generation = nextGeneration.fetch_add(1, std::memory_order_relaxed);

    g.verts.clear();
    g.flightVerts.clear();
    g.bodyVerts.clear();
    auto append = [](const DartPartMesh& m, std::vector<glm::vec3>& dst) {
        for (size_t i = 0; i + 2 < m.vertices.size(); i += kFloats)
            dst.emplace_back(m.vertices[i], m.vertices[i + 1],
                             m.vertices[i + 2]);
    };
    append(b.point, g.bodyVerts);
    append(b.barrel, g.bodyVerts);
    // The whole tail counts as flight, stem included. The stem runs on inside
    // the vanes, so splitting it off by z would let two shafts overlap
    // undetected -- and the vanes are what decide roll-dependent collisions
    // either way.
    append(b.tail, g.flightVerts);
    g.verts = g.bodyVerts;
    g.verts.insert(g.verts.end(), g.flightVerts.begin(), g.flightVerts.end());
}

} // namespace dart
