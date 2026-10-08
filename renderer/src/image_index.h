#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace dart {

// The image files in one directory, listed once and sampled uniformly.
//
// Backgrounds are drawn at random rather than walked in directory order.
// Places365 is extracted to sequential names in category order, so a walk
// would show one or two scene categories per epoch, and every renderer would
// walk the same sequence from the same start.
//
// Names are sorted after listing, because directory order differs between
// filesystems: a given seed then draws the same images on every machine.
// They are stored packed in one buffer -- the full Places365 train split is
// ~1.8M files, about 30 MB this way against several times that as strings.
class ImageIndex {
public:
    /// List the image files (jpg, jpeg, png, bmp, webp) directly in `dir`.
    /// Returns false if `dir` is not a directory.
    bool scan(const std::string& dir);

    size_t size() const { return offsets_.empty() ? 0 : offsets_.size() - 1; }

    /// Full path of image `i`, in sorted order.
    std::string path(size_t i) const;

    /// A uniformly random image.
    std::string sample(std::mt19937& rng) const;

private:
    std::string dir_;
    std::string names_;              ///< every file name, concatenated
    std::vector<uint32_t> offsets_;  ///< name i is names_[offsets_[i], offsets_[i+1])
};

} // namespace dart
