#include "image_index.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace dart {

static bool isImageExt(const fs::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
           ext == ".webp";
}

bool ImageIndex::scan(const std::string& dir) {
    names_.clear();
    offsets_.clear();
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return false;
    dir_ = dir;

    std::vector<std::string> names;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (entry.is_regular_file(ec) && isImageExt(entry.path())) {
            names.push_back(entry.path().filename().string());
        }
    }
    std::sort(names.begin(), names.end());

    size_t total = 0;
    for (const auto& n : names) total += n.size();
    if (total > UINT32_MAX) throw std::runtime_error("ImageIndex: too many names");
    names_.reserve(total);
    offsets_.reserve(names.size() + 1);
    offsets_.push_back(0);
    for (const auto& n : names) {
        names_ += n;
        offsets_.push_back((uint32_t)names_.size());
    }
    return true;
}

std::string ImageIndex::path(size_t i) const {
    const std::string name = names_.substr(offsets_[i], offsets_[i + 1] - offsets_[i]);
    return (fs::path(dir_) / name).string();
}

std::string ImageIndex::sample(std::mt19937& rng) const {
    if (size() == 0) return {};
    std::uniform_int_distribution<size_t> pick(0, size() - 1);
    return path(pick(rng));
}

} // namespace dart
