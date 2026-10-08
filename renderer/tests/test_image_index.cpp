// Background photos are drawn at random, reproducibly per seed.
//
// Places365 is extracted to sequential names in category order, so reading
// the directory in order would show a couple of scene categories per epoch,
// the same ones for every renderer. These pin the sampling instead: images
// come from the whole directory, a seed reproduces its draws, and different
// seeds draw differently.

#include "image_index.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using dart::ImageIndex;

static int failures = 0;

#define CHECK(cond, ...)                                         \
    do {                                                         \
        if (!(cond)) {                                           \
            ++failures;                                          \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                   \
            std::fprintf(stderr, "\n");                          \
        }                                                        \
    } while (0)

static std::vector<std::string> draws(const ImageIndex& index, uint32_t seed, int n) {
    std::mt19937 rng(seed);
    std::vector<std::string> out;
    for (int i = 0; i < n; ++i) out.push_back(index.sample(rng));
    return out;
}

int main() {
    const fs::path dir = fs::temp_directory_path() / "darts_image_index_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const int kImages = 1000;
    for (int i = 0; i < kImages; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "%07d.jpg", i);
        std::ofstream(dir / name) << "x";
    }
    std::ofstream(dir / "notes.txt") << "not an image";
    std::ofstream(dir / "UPPER.PNG") << "x";
    fs::create_directories(dir / "subdir.jpg");

    ImageIndex index;
    CHECK(index.scan(dir.string()), "scan failed");
    CHECK(index.size() == kImages + 1, "expected %d images, got %zu",
          kImages + 1, index.size());
    CHECK(fs::path(index.path(0)).filename() == "0000000.jpg",
          "not sorted: first is %s", index.path(0).c_str());

    ImageIndex missing;
    CHECK(!missing.scan((dir / "does_not_exist").string()), "scanned a missing dir");
    CHECK(missing.size() == 0, "missing dir has images");

    const int kDraws = 2000;
    const auto a1 = draws(index, 1, kDraws);
    const auto a2 = draws(index, 1, kDraws);
    const auto b = draws(index, 2, kDraws);
    CHECK(a1 == a2, "same seed drew different images");
    CHECK(a1 != b, "different seeds drew the same images");

    // Not a walk: consecutive draws are rarely neighbours in sorted order.
    int neighbours = 0;
    for (int i = 1; i < kDraws; ++i) {
        const std::string prev = fs::path(a1[i - 1]).stem().string();
        const std::string cur = fs::path(a1[i]).stem().string();
        if (prev.size() == 7 && cur.size() == 7 &&
            std::abs(std::atoi(cur.c_str()) - std::atoi(prev.c_str())) <= 1)
            ++neighbours;
    }
    CHECK(neighbours < kDraws / 50, "%d of %d consecutive draws were neighbours",
          neighbours, kDraws);

    // Covers the directory: 2000 uniform draws from 1001 reach ~86% of it.
    const std::set<std::string> seen(a1.begin(), a1.end());
    CHECK(seen.size() > (size_t)(0.75 * index.size()),
          "only %zu of %zu images drawn", seen.size(), index.size());

    fs::remove_all(dir);
    if (failures == 0) std::printf("test_image_index: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
