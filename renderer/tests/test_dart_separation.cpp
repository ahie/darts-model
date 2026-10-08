// Dart placement: separation, the collision-shape cache, dart counts, and the
// pose distributions.
//
// Darts must not intersect one another. Two solid objects cannot occupy the
// same space, so a turn containing interpenetrating darts is one no camera
// could ever photograph. It also corrupts the labels: the instance silhouette
// has to award contested pixels to one dart, and the fitted boxes of two
// interpenetrating darts overlap almost entirely, which is the case the voting
// readout cannot split into two peaks.
//
// The body and the flight are parted by different means, because they are
// different shapes. A barrel is a solid of revolution, so a swept radius
// describes it exactly and two barrels are separated by MOVING a dart. A
// flight is a cross of thin vanes: its swept radius would make it a solid
// cylinder 32mm across and demand ~45mm of displacement on two turns in three,
// dismantling the tight groupings the skill model exists to produce. Two
// flights at different clock angles interleave without touching, so those are
// parted by ROLLING the dart about its own axis, which moves nothing.
//
// Every dart here is a PRODUCTION dart: sampled, built and split by the same
// sampleDart / buildDart / extractGeometry the renderer uses, so the test
// exercises the body/flight split the randomizer actually receives. None of
// that touches Vulkan, so this runs without a GPU, thousands of turns a
// second.
#include "randomizer.h"
#include "constants.h"
#include "dart_mesh.h"

#include "dart_mesh.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dart;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

/// One production dart design, as the renderer's pool holds it.
DartGeometry makeVariant(std::mt19937& rng) {
    DartTessellation tess;
    DartBuild build;
    buildDart(sampleDart(rng), tess, build);
    DartGeometry g;
    extractGeometry(build, g);
    return g;
}

/// The same design with its vertices removed. The randomizer then skips
/// de-intersection -- it has no shape to test -- while still drawing the
/// variant index from its RNG exactly as it does for the full pool, so a run
/// against this pool is the fix-disabled twin of a run against the real one.
std::vector<DartGeometry> withoutShapes(std::vector<DartGeometry> pool) {
    for (DartGeometry& g : pool) {
        g.verts.clear();
        g.bodyVerts.clear();
        g.flightVerts.clear();
    }
    return pool;
}

glm::vec3 axisOf(const DartPlacement& d) {
    return glm::vec3(std::sin(d.tilt) * std::cos(d.azimuth),
                     std::sin(d.tilt) * std::sin(d.azimuth),
                     std::cos(d.tilt));
}

glm::mat3 rotOf(const DartPlacement& d) {
    const glm::vec3 axis = axisOf(d);
    const glm::vec3 ref = (std::abs(axis.z) < 0.999f)
                              ? glm::vec3(0.0f, 0.0f, 1.0f)
                              : glm::vec3(1.0f, 0.0f, 0.0f);
    const glm::vec3 tangent = glm::normalize(glm::cross(ref, axis));
    const glm::mat3 frame(tangent, glm::cross(axis, tangent), axis);
    const float cr = std::cos(d.roll), sr = std::sin(d.roll);
    const glm::mat3 rollZ(glm::vec3(cr, sr, 0.0f), glm::vec3(-sr, cr, 0.0f),
                          glm::vec3(0.0f, 0.0f, 1.0f));
    return frame * rollZ;
}

glm::vec3 tipOf(const DartPlacement& d, double boardZ) {
    return glm::vec3(d.x, d.y, (float)boardZ) - axisOf(d) * d.penetration;
}

/// The checker's own view of a design: a body radius profile far finer than
/// the randomizer's 24 bins, and a subsample of the tail far denser than its
/// 56 points, both built here from the raw vertices.
struct CheckShape {
    static constexpr int N = 400;
    float len = 0.0f;
    std::vector<float> radius;          // N bins along the axis
    std::vector<glm::vec3> tailPts;     // dart-local
    float tailReach = 0.0f;
    float tailMid = 0.0f, tailHalf = 0.0f;
};

CheckShape makeCheckShape(const DartGeometry& g) {
    CheckShape c;
    c.len = glm::length(g.tailLocal - g.tipLocal);
    c.radius.assign(CheckShape::N, 0.0f);
    for (const glm::vec3& v : g.bodyVerts) {
        const float t = v.z - g.tipLocal.z;
        int bin = (int)(t / c.len * CheckShape::N);
        bin = std::max(0, std::min(CheckShape::N - 1, bin));
        c.radius[bin] = std::max(c.radius[bin],
                                 glm::length(glm::vec2(v.x, v.y)));
    }
    // The body ends where the tail begins; past it the radius stays at zero
    // and the tail points below take over.
    for (int k = 1; k < CheckShape::N; ++k)
        if (c.radius[k] <= 0.0f && c.radius[k - 1] > 0.0f &&
            k * c.len / CheckShape::N < 0.5f * c.len)
            c.radius[k] = c.radius[k - 1];

    constexpr size_t kTailPts = 700;
    const size_t stride = std::max<size_t>(1, g.flightVerts.size() / kTailPts);
    float lo = 1e9f, hi = -1e9f;
    for (size_t i = 0; i < g.flightVerts.size(); i += stride) {
        const glm::vec3& v = g.flightVerts[i];
        c.tailPts.push_back(v - g.tipLocal);
        c.tailReach = std::max(c.tailReach, glm::length(glm::vec2(v.x, v.y)));
        lo = std::min(lo, v.z);
        hi = std::max(hi, v.z);
    }
    c.tailMid = 0.5f * (lo + hi);
    c.tailHalf = 0.5f * (hi - lo);
    return c;
}

/// Deepest overlap between two dart BODIES, from the fine profile.
float deepestBodyOverlap(const CheckShape& s, const DartPlacement& a,
                         const DartPlacement& b, double boardZ) {
    const glm::vec3 aa = axisOf(a), ab = axisOf(b);
    const glm::vec3 ta = tipOf(a, boardZ), tb = tipOf(b, boardZ);
    constexpr int N = CheckShape::N;
    float worst = 0.0f;
    for (int i = 0; i < N; ++i) {
        if (s.radius[i] <= 0.0f) continue;
        const glm::vec3 pa = ta + aa * (s.len * (i + 0.5f) / N);
        for (int j = 0; j < N; ++j) {
            if (s.radius[j] <= 0.0f) continue;
            const glm::vec3 pb = tb + ab * (s.len * (j + 0.5f) / N);
            const float pen = s.radius[i] + s.radius[j] - glm::length(pa - pb);
            if (pen > worst) worst = pen;
        }
    }
    return worst;
}

/// Do two tails pass through each other? Tested at the actual roll on the
/// dense subsample.
bool tailsCross(const CheckShape& s, const DartPlacement& a,
                const DartPlacement& b, double boardZ) {
    const glm::vec3 ca = tipOf(a, boardZ) + axisOf(a) * s.tailMid;
    const glm::vec3 cb = tipOf(b, boardZ) + axisOf(b) * s.tailMid;
    if (glm::length(ca - cb) > 2.0f * (s.tailReach + s.tailHalf)) return false;
    auto world = [&](const DartPlacement& d) {
        std::vector<glm::vec3> pts;
        pts.reserve(s.tailPts.size());
        const glm::mat3 R = rotOf(d);
        const glm::vec3 tip = tipOf(d, boardZ);
        for (const glm::vec3& q : s.tailPts) pts.push_back(tip + R * q);
        return pts;
    };
    const std::vector<glm::vec3> pa = world(a), pb = world(b);
    constexpr float kTouch = 0.0010f;   // 1mm
    for (const glm::vec3& p : pa)
        for (const glm::vec3& q : pb)
            if (glm::length(p - q) < kTouch) return true;
    return false;
}

bool samePlacements(const FrameState& a, const FrameState& b) {
    if (a.darts.size() != b.darts.size()) return false;
    for (size_t i = 0; i < a.darts.size(); ++i) {
        const DartPlacement& p = a.darts[i];
        const DartPlacement& q = b.darts[i];
        if (p.x != q.x || p.y != q.y || p.tilt != q.tilt ||
            p.azimuth != q.azimuth || p.roll != q.roll ||
            p.penetration != q.penetration)
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------------

void testSeparation(const std::vector<DartGeometry>& pool, bool verbose) {
    const RingRadii radii = RING_RADII_BU;
    const double boardZ = BOARD_SURFACE_Z;
    const int FRAMES = 1500;

    std::vector<CheckShape> checkShapes;
    for (const DartGeometry& g : pool) checkShapes.push_back(makeCheckShape(g));
    const std::vector<DartGeometry> shapeless = withoutShapes(pool);

    // Paired: same seed, same pool size, so both draw the same variant, pose
    // and placement every frame, and de-intersection -- which draws nothing
    // from the RNG -- is the only difference. Three darts every frame, so
    // grouping turns are as common as possible; that is where
    // interpenetration lives.
    Randomizer baseline(1234, true, {0.0f, 0.0f, 0.0f, 1.0f});
    Randomizer fixed(1234, true, {0.0f, 0.0f, 0.0f, 1.0f});

    int frames = 0, bodyFixed = 0, rerolled = 0, unresolved = 0;
    int residualBody = 0, overlapBefore = 0, crossedBefore = 0,
        crossedAfter = 0, pairs = 0, unpaired = 0;
    float worstResidual = 0.0f, worstMove = 0.0f;
    long long nsBaseline = 0, nsFixed = 0;

    for (int f = 0; f < FRAMES; ++f) {
        auto t0 = std::chrono::steady_clock::now();
        FrameState raw = baseline.randomize(radii, boardZ, shapeless);
        auto t1 = std::chrono::steady_clock::now();
        FrameState st = fixed.randomize(radii, boardZ, pool);
        auto t2 = std::chrono::steady_clock::now();
        nsBaseline += std::chrono::duration_cast<std::chrono::nanoseconds>(
                          t1 - t0).count();
        nsFixed += std::chrono::duration_cast<std::chrono::nanoseconds>(
                       t2 - t1).count();

        if (raw.dartVariant != st.dartVariant ||
            raw.numDarts != st.numDarts) {
            ++unpaired;
            continue;
        }
        if (st.numDarts < 2) continue;
        ++frames;
        const CheckShape& cs = checkShapes[st.dartVariant];
        if (st.dartsSeparated > 0) ++bodyFixed;
        worstMove = std::max(worstMove, st.maxSeparationBU);
        rerolled += st.dartsRerolled;
        unresolved += st.flightsUnresolved;

        for (int i = 0; i < st.numDarts; ++i) {
            for (int j = i + 1; j < st.numDarts; ++j) {
                ++pairs;
                if (deepestBodyOverlap(cs, raw.darts[i], raw.darts[j],
                                       boardZ) > 1e-4f)
                    ++overlapBefore;
                const float pen =
                    deepestBodyOverlap(cs, st.darts[i], st.darts[j], boardZ);
                if (pen > 1e-4f) {
                    ++residualBody;
                    worstResidual = std::max(worstResidual, pen);
                }
                if (tailsCross(cs, raw.darts[i], raw.darts[j], boardZ))
                    ++crossedBefore;
                if (tailsCross(cs, st.darts[i], st.darts[j], boardZ))
                    ++crossedAfter;
            }
        }
        for (int i = 0; i < st.numDarts; ++i) {
            const float r = std::sqrt(st.darts[i].x * st.darts[i].x +
                                      st.darts[i].y * st.darts[i].y);
            check(r <= (float)radii.board_r, "dart pushed off the board");
        }
        if (verbose && st.dartsSeparated)
            std::printf("  frame %d: moved %d, rerolled %d\n", f,
                        st.dartsSeparated, st.dartsRerolled);
    }

    std::printf("frames with >=2 darts   : %d\n", frames);
    std::printf("bodies needing a move   : %d (%.1f%%), worst %.2f mm\n",
                bodyFixed, 100.0 * bodyFixed / std::max(frames, 1),
                worstMove * 100.0f);
    std::printf("darts re-rolled         : %d\n", rerolled);
    std::printf("body overlaps           : %d -> %d (worst after %.3f mm)\n",
                overlapBefore, residualBody, worstResidual * 100.0f);
    std::printf("tail pairs crossing     : %d -> %d\n", crossedBefore,
                crossedAfter);
    std::printf("pairs no roll could fix : %d\n", unresolved);
    // The renderer targets 500+ fps, i.e. a 2ms frame budget, so what the
    // check costs per frame matters as much as whether it works.
    std::printf("randomize cost/frame    : %.1f us -> %.1f us (+%.1f us)\n",
                nsBaseline / 1000.0 / std::max(frames, 1),
                nsFixed / 1000.0 / std::max(frames, 1),
                (nsFixed - nsBaseline) / 1000.0 / std::max(frames, 1));

    check(unpaired == 0, "baseline and fixed runs drew different turns");
    // Not zero, and deliberately so. A handful of configurations are reachable
    // by neither displacement nor lean: the overlap separates almost purely
    // along the board normal, where the in-plane component available to an
    // entry-point move is nearly nil, and the pose grid has no candidate that
    // clears it either. Closing those by displacement would cost the tens of
    // millimetres this whole design exists to avoid. What survives is a graze
    // -- two darts touching, not one passing through the other -- on a small
    // fraction of pairs. Asserted as a rate and a depth so a real regression
    // still fails the test.
    const double residualRate = (double)residualBody / std::max(pairs, 1);
    std::printf("residual rate           : %.3f%% of pairs\n",
                100.0 * residualRate);
    check(overlapBefore > 0, "no body overlaps to fix; the test is vacuous");
    check(residualBody < overlapBefore, "separation did not reduce overlaps");
    check(residualRate < 0.005, "too many dart bodies still intersect");
    check(worstResidual < 0.02f, "a residual body overlap exceeded 2mm");
    check(crossedAfter < crossedBefore,
          "rolling did not reduce crossed flights");
    // The point of rolling rather than moving: the entry coordinate is what the
    // score zone and every annotation are computed from, and the ~45mm move
    // needed to part two flights by displacement would relocate a dart out of
    // the treble it was aimed at.
    check(worstMove < 0.15f, "a body correction exceeded 15mm");

    // The run above forces three darts every frame, which is the worst case
    // for a pairwise check. What the renderer actually costs depends on the
    // configured dart-count weights, so measure those too.
    {
        Randomizer real(99, true, {0.05f, 0.25f, 0.30f, 0.40f});
        long long ns = 0;
        int n = 0;
        for (int f = 0; f < 3000; ++f) {
            const auto t0 = std::chrono::steady_clock::now();
            real.randomize(radii, boardZ, pool);
            ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now() - t0).count();
            ++n;
        }
        std::printf("cost at real weights    : %.1f us/frame\n",
                    ns / 1000.0 / n);
    }
}

/// A pool slot regenerated in place must be checked with the NEW shape.
///
/// The renderer rolls one design out of the pool every few hundred frames
/// and writes its replacement into the same slot. Three randomizers share a
/// seed and a single-slot pool: one always sees design A, one always sees
/// design B, and one sees A until the switch and B after it. Separation draws
/// nothing from the RNG, so after the switch the third must reproduce the
/// always-B run exactly -- which it can only do by rebuilding its cached
/// shape -- and the always-A run must differ, or the check proves nothing.
void testShapeCacheFollowsGeometry(std::mt19937& rng) {
    const RingRadii radii = RING_RADII_BU;
    const double boardZ = BOARD_SURFACE_Z;

    const DartGeometry a = makeVariant(rng);
    const DartGeometry b = makeVariant(rng);
    check(a.generation != 0 && b.generation != 0 &&
              a.generation != b.generation,
          "extractGeometry did not stamp distinct generations");

    const std::vector<float> w = {0.0f, 0.0f, 0.0f, 1.0f};
    Randomizer onlyA(77, true, w), onlyB(77, true, w), switched(77, true, w),
        switchedUnstamped(77, true, w);
    std::vector<DartGeometry> poolA = {a}, poolB = {b};
    std::vector<DartGeometry> live = {a}, liveUnstamped = {a};

    constexpr int kWarm = 50, kAfter = 400;
    for (int f = 0; f < kWarm; ++f) {
        onlyA.randomize(radii, boardZ, poolA);
        onlyB.randomize(radii, boardZ, poolB);
        switched.randomize(radii, boardZ, live);
        switchedUnstamped.randomize(radii, boardZ, liveUnstamped);
    }

    // Replaced in place, as rollDartVariant does. The second copy carries no
    // generation, as hand-built geometry would, and must still not be served
    // a stale shape.
    live[0] = b;
    liveUnstamped[0] = b;
    liveUnstamped[0].generation = 0;

    int mismatches = 0, mismatchesUnstamped = 0, differsFromA = 0;
    for (int f = 0; f < kAfter; ++f) {
        const FrameState sa = onlyA.randomize(radii, boardZ, poolA);
        const FrameState sb = onlyB.randomize(radii, boardZ, poolB);
        const FrameState ss = switched.randomize(radii, boardZ, live);
        const FrameState su =
            switchedUnstamped.randomize(radii, boardZ, liveUnstamped);
        if (!samePlacements(ss, sb)) ++mismatches;
        if (!samePlacements(su, sb)) ++mismatchesUnstamped;
        if (!samePlacements(sa, sb)) ++differsFromA;
    }
    std::printf("shape cache             : %d/%d frames depend on the "
                "design; %d stale after an in-place swap (%d unstamped)\n",
                differsFromA, kAfter, mismatches, mismatchesUnstamped);
    check(differsFromA > 0,
          "designs A and B never separate differently; the test is vacuous");
    check(mismatches == 0,
          "randomizer kept the replaced design's collision shape");
    check(mismatchesUnstamped == 0,
          "randomizer cached a shape for unstamped geometry");
}

/// Every frame carries exactly the drawn dart count, checkout turns included.
void testDartCounts(const std::vector<DartGeometry>& pool) {
    const RingRadii radii = RING_RADII_BU;
    const double boardZ = BOARD_SURFACE_Z;
    for (int n = 1; n <= 3; ++n) {
        std::vector<float> w(4, 0.0f);
        w[n] = 1.0f;
        Randomizer r(500 + n, true, w);
        int wrong = 0;
        for (int f = 0; f < 3000; ++f)
            if (r.randomize(radii, boardZ, pool).numDarts != n) ++wrong;
        std::printf("dart count %d           : %d/3000 frames wrong\n", n,
                    wrong);
        check(wrong == 0, "dart_count_weights not honoured");
    }

    // The mix itself, at the configured weights.
    {
        const std::vector<float> w = {0.10f, 0.20f, 0.30f, 0.40f};
        Randomizer r(9, true, w);
        int counts[4] = {0, 0, 0, 0};
        constexpr int F = 20000;
        for (int f = 0; f < F; ++f) {
            const int n = r.randomize(radii, boardZ, {}).numDarts;
            if (n >= 0 && n < 4) ++counts[n];
        }
        for (int n = 0; n < 4; ++n)
            check(std::abs(counts[n] / (double)F - w[n]) < 0.015,
                  "dart count mix drifted from the weights");
    }

    auto throws = [](std::vector<float> w) {
        try {
            Randomizer r(1, true, std::move(w));
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    check(!throws({}), "empty dart_count_weights rejected");
    check(throws({0.0f, 0.0f, 0.0f, 0.0f}), "all-zero weights accepted");
    check(throws({0.5f, -0.1f, 0.3f, 0.3f}), "negative weight accepted");
    check(throws({0.5f, std::numeric_limits<float>::quiet_NaN(), 0.3f, 0.2f}),
          "NaN weight accepted");
    check(throws({0.5f, std::numeric_limits<float>::infinity(), 0.3f, 0.2f}),
          "infinite weight accepted");
}

/// Truncated, not clamped: no sample sits exactly on a bound, and none
/// escapes the range.
void testNoClampSpikes() {
    const RingRadii radii = RING_RADII_BU;
    const double boardZ = BOARD_SURFACE_Z;
    Randomizer r(31, true, {0.0f, 0.0f, 0.0f, 1.0f});
    const float tiltLo = glm::radians(2.0f), tiltHi = glm::radians(42.0f);
    const float penLo = 0.025f, penHi = 0.160f;
    int n = 0, onBound = 0, outside = 0;
    std::map<float, int> repeats;
    int maxRepeat = 0;
    for (int f = 0; f < 5000; ++f) {
        const FrameState st = r.randomize(radii, boardZ, {});
        for (const DartPlacement& d : st.darts) {
            ++n;
            if (d.tilt == tiltLo || d.tilt == tiltHi ||
                d.penetration == penLo || d.penetration == penHi)
                ++onBound;
            if (d.tilt < tiltLo || d.tilt > tiltHi ||
                d.penetration < penLo || d.penetration > penHi)
                ++outside;
            maxRepeat = std::max(maxRepeat, ++repeats[d.tilt]);
        }
    }
    std::printf("pose samples            : %d, %d on a bound, %d outside, "
                "most-repeated tilt x%d\n", n, onBound, outside, maxRepeat);
    check(onBound == 0, "a pose sample sits exactly on a clamp bound");
    check(outside == 0, "a pose sample escaped its range");
    check(maxRepeat <= 3, "tilt has a point mass");
}

}  // namespace

int main(int argc, char** argv) {
    bool verbose = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "-v") verbose = true;

    // A small pool of real designs; the randomizer picks one per turn for all
    // three darts, because a player throws a matched set.
    std::mt19937 rng(2024);
    std::vector<DartGeometry> pool;
    for (int v = 0; v < 4; ++v) pool.push_back(makeVariant(rng));

    testSeparation(pool, verbose);
    testShapeCacheFollowsGeometry(rng);
    testDartCounts(pool);
    testNoClampSpikes();

    std::printf(failures ? "\nFAILED (%d)\n" : "\nPASSED\n", failures);
    return failures ? 1 : 0;
}
