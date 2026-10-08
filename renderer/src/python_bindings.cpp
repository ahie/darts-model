#include "renderer_lib.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// JSON value -> Python object. The annotation is serialised once, by
// annotationToJson, and converted here, so the dict training sees and the
// files dartboard_gen writes cannot drift apart.
static py::object jsonToPy(const nlohmann::json& j) {
    switch (j.type()) {
    case nlohmann::json::value_t::null:
        return py::none();
    case nlohmann::json::value_t::boolean:
        return py::bool_(j.get<bool>());
    case nlohmann::json::value_t::number_integer:
        return py::int_(j.get<int64_t>());
    case nlohmann::json::value_t::number_unsigned:
        return py::int_(j.get<uint64_t>());
    case nlohmann::json::value_t::number_float:
        return py::float_(j.get<double>());
    case nlohmann::json::value_t::string:
        return py::str(j.get_ref<const std::string&>());
    case nlohmann::json::value_t::array: {
        py::list out;
        for (const auto& v : j) out.append(jsonToPy(v));
        return out;
    }
    case nlohmann::json::value_t::object: {
        py::dict out;
        for (auto it = j.begin(); it != j.end(); ++it)
            out[py::str(it.key())] = jsonToPy(it.value());
        return out;
    }
    default:
        throw std::runtime_error("annotation JSON holds an unsupported type");
    }
}

PYBIND11_MODULE(dartboard_renderer, m) {
    m.doc() = "Vulkan dartboard renderer — Python bindings";

    py::class_<dart::Renderer>(m, "Renderer")
        .def(py::init<const std::string&, int, int, uint32_t, int, const std::string&, bool, std::vector<float>, float, float, float, float, float, const std::string&>(),
             py::arg("asset_dir") = std::string(DEFAULT_ASSET_DIR),
             py::arg("width") = 1024,
             py::arg("height") = 1024,
             py::arg("seed") = 42,
             py::arg("gpu_index") = -1,
             py::arg("bg_image_dir") = "",
             py::arg("skill_placement") = true,
             py::arg("dart_count_weights") = std::vector<float>(),
             py::arg("grouping_prob") = 0.55f,
             py::arg("tight_prob") = 0.55f,
             // Camera orbit half-widths. 60/48 covers the obliquity measured
             // over 49 real captures (median 42.8 deg, max 67); a 45/35 orbit
             // tops out at 54.5 deg, which would leave 29% of real frames
             // outside anything the renderer can draw.
             py::arg("pan_max_deg") = 60.0f,
             py::arg("tilt_max_deg") = 48.0f,
             // Scales the ceiling fixture's radiance (see env_room.glsl).
             // 0 leaves the bare photo probe with no fixture.
             py::arg("env_room_scale") = 1.0f,
             // Selects the GPU by UUID (as torch.cuda reports it) and takes
             // precedence over gpu_index.
             py::arg("gpu_uuid") = "")
        .def_property_readonly("width", &dart::Renderer::width)
        .def_property_readonly("height", &dart::Renderer::height)
        .def_property_readonly("has_bg_textures", &dart::Renderer::hasBgTextures)
        .def("render_frame", [](dart::Renderer& self) {
            dart::Renderer::FrameResult result;

            // Release GIL during GPU rendering
            {
                py::gil_scoped_release release;
                result = self.renderFrame();
            }

            int h = self.height();
            int w = self.width();
            int ch = result.channels;

            // Create numpy array that owns the pixel data (zero-copy)
            auto* pixels = new std::vector<uint8_t>(std::move(result.pixels));
            auto capsule = py::capsule(pixels, [](void* p) {
                delete static_cast<std::vector<uint8_t>*>(p);
            });
            py::array_t<uint8_t> image(
                {h, w, ch},                          // shape
                {w * ch, ch, 1},                      // strides (row-major)
                pixels->data(),                       // data pointer
                capsule                               // prevent deallocation
            );

            py::dict annotation =
                jsonToPy(dart::annotationToJson(result.annotation))
                    .cast<py::dict>();

            // Dense labels ride in the annotation dict rather than widening the
            // return tuple, so render_frame() keeps returning (image, ann).
            // They are arrays, so they are the one part of the annotation that
            // is not in the JSON.
            if (!result.depth.empty()) {
                auto* dep = new std::vector<float>(std::move(result.depth));
                auto depCapsule = py::capsule(dep, [](void* p) {
                    delete static_cast<std::vector<float>*>(p);
                });
                annotation["depth"] = py::array_t<float>(
                    {h, w}, {w * (int)sizeof(float), (int)sizeof(float)},
                    dep->data(), depCapsule);
            }

            if (!result.segIds.empty()) {
                auto* seg = new std::vector<uint8_t>(std::move(result.segIds));
                auto segCapsule = py::capsule(seg, [](void* p) {
                    delete static_cast<std::vector<uint8_t>*>(p);
                });
                annotation["seg_ids"] = py::array_t<uint8_t>(
                    {h, w, 2}, {w * 2, 2, 1}, seg->data(), segCapsule);
            }

            if (!result.segSubpixel.empty()) {
                auto* sub = new std::vector<uint8_t>(std::move(result.segSubpixel));
                auto subCapsule = py::capsule(sub, [](void* p) {
                    delete static_cast<std::vector<uint8_t>*>(p);
                });
                annotation["seg_subpixel"] = py::array_t<uint8_t>(
                    {h, w}, {w, 1}, sub->data(), subCapsule);
                annotation["supersample"] = dart::kSupersample;
            }

            return py::make_tuple(image, annotation);
        }, "Render one frame and return (image, annotation) tuple.\n\n"
           "image: numpy array (H, W, C) uint8 — RGB (3ch) when bg_image_dir is set,\n"
           "       RGBA (4ch) otherwise (alpha=0 background, 255 geometry)\n"
           "annotation: dict with the keys of annotationToJson (board_keypoints,\n"
           "       darts, camera_params, metadata), plus depth, a (H, W) float32\n"
           "       window-space depth, and seg_ids, a (H, W, 2) uint8 array whose\n"
           "       channels are dart::SegClass and the dart instance (0 = not a\n"
           "       dart, else dart index + 1), and seg_subpixel, (H, W) uint8,\n"
           "       the subsample dy * supersample + dx that seg_ids and depth\n"
           "       were taken from");
}
