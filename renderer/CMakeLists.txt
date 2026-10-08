cmake_minimum_required(VERSION 3.21)
project(dartboard_gen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)

# MSVC: enable M_PI and other POSIX math constants
if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
find_package(Vulkan REQUIRED)
if(WIN32)
    find_package(libjpeg-turbo CONFIG REQUIRED)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(TURBOJPEG REQUIRED IMPORTED_TARGET libturbojpeg)
endif()

# ---------------------------------------------------------------------------
# FetchContent dependencies
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    vk-bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG        v1.3.296
)

FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.1.0
)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)

FetchContent_MakeAvailable(vk-bootstrap VulkanMemoryAllocator glm nlohmann_json pybind11)

# Force PIC on vk-bootstrap and suppress warnings from third-party code
if(TARGET vk-bootstrap)
    set_target_properties(vk-bootstrap PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        COMPILE_OPTIONS "-w"
    )
endif()

# ---------------------------------------------------------------------------
# Single-header libraries (vendored in third_party/)
# ---------------------------------------------------------------------------
# One target carries both the implementation unit and, PUBLIC, the include
# path, so linking it is all a consumer needs.
add_library(third_party_impl STATIC third_party/third_party_impl.cpp)
target_include_directories(third_party_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# ---------------------------------------------------------------------------
# Shader compilation (SPIR-V)
# ---------------------------------------------------------------------------
find_program(GLSLC glslc glslangValidator HINTS $ENV{VULKAN_SDK}/bin)
if(NOT GLSLC)
    message(FATAL_ERROR "glslc or glslangValidator not found — install Vulkan SDK or glslang-tools")
endif()
# Real glslc (shaderc) always prints "shaderc" in --version.
# If missing, the binary is glslangValidator (or a symlink to it) and needs -V.
execute_process(
    COMMAND ${GLSLC} --version
    OUTPUT_VARIABLE _GLSLC_VER ERROR_VARIABLE _GLSLC_VER
)
if(NOT _GLSLC_VER MATCHES "shaderc")
    set(GLSLC_IS_GLSLANG TRUE)
endif()

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SPIRV_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SPIRV_DIR})

set(SHADER_SOURCES
    ${SHADER_DIR}/surface.vert
    ${SHADER_DIR}/surface_raster.frag
    ${SHADER_DIR}/surface_rt.frag
    ${SHADER_DIR}/background.vert
    ${SHADER_DIR}/background.frag
    ${SHADER_DIR}/segment.vert
    ${SHADER_DIR}/segment.frag
)

# Headers pulled in via #include. Listed as explicit dependencies because the
# compile rule only knows about the .frag it is handed -- without this, editing
# a shared shader header rebuilds nothing.
set(SHADER_INCLUDES
    ${SHADER_DIR}/board_face.glsl
    ${SHADER_DIR}/env_room.glsl
    ${SHADER_DIR}/surface_common.glsl
)

set(SPIRV_OUTPUTS)
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPIRV_OUT ${SPIRV_DIR}/${SHADER_NAME}.spv)
    if(GLSLC_IS_GLSLANG)
        add_custom_command(
            OUTPUT ${SPIRV_OUT}
            COMMAND ${GLSLC} -V --target-env vulkan1.2 -I${SHADER_DIR} ${SHADER} -o ${SPIRV_OUT}
            DEPENDS ${SHADER} ${SHADER_INCLUDES}
            COMMENT "Compiling ${SHADER_NAME} -> SPIR-V (glslangValidator)"
        )
    else()
        add_custom_command(
            OUTPUT ${SPIRV_OUT}
            COMMAND ${GLSLC} --target-env=vulkan1.2 -I${SHADER_DIR} ${SHADER} -o ${SPIRV_OUT}
            DEPENDS ${SHADER} ${SHADER_INCLUDES}
            COMMENT "Compiling ${SHADER_NAME} -> SPIR-V"
        )
    endif()
    list(APPEND SPIRV_OUTPUTS ${SPIRV_OUT})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPIRV_OUTPUTS})

# Compiled SPIR-V is read from the build tree at runtime, and the numeral GLBs
# and decal atlas from the source tree unless --asset_dir / asset_dir says
# otherwise. Both paths are fixed at configure time.
add_compile_definitions(
    SHADER_DIR="${SPIRV_DIR}"
    DEFAULT_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
)

# ---------------------------------------------------------------------------
# Core renderer sources (shared between CLI and Python module)
# ---------------------------------------------------------------------------
set(RENDERER_CORE_SOURCES
    src/renderer_lib.cpp
    src/vk_context.cpp
    src/render_pass.cpp
    src/scene.cpp
    src/accel_structure.cpp
    src/randomizer.cpp
    src/annotation.cpp
    src/frame_pipeline.cpp
    src/wire_geometry.cpp
    src/board_mesh.cpp
    src/dart_mesh.cpp
    src/image_index.cpp
)

set(RENDERER_CORE_LIBS
    Vulkan::Vulkan
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    glm::glm
    nlohmann_json::nlohmann_json
    $<IF:$<BOOL:${WIN32}>,libjpeg-turbo::turbojpeg,PkgConfig::TURBOJPEG>
    third_party_impl
)

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------
add_executable(dartboard_gen
    src/main.cpp
    ${RENDERER_CORE_SOURCES}
)

add_dependencies(dartboard_gen shaders)

target_include_directories(dartboard_gen PRIVATE src)

target_link_libraries(dartboard_gen PRIVATE ${RENDERER_CORE_LIBS})

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# The randomizer depends only on glm, so dart placement can be exercised
# without a Vulkan context -- no GPU, no assets, thousands of turns a second.
add_executable(test_dart_separation
    tests/test_dart_separation.cpp
    src/randomizer.cpp
    src/dart_mesh.cpp
)
target_include_directories(test_dart_separation PRIVATE src)
target_link_libraries(test_dart_separation PRIVATE glm)

add_executable(test_image_index
    tests/test_image_index.cpp
    src/image_index.cpp
)
target_include_directories(test_image_index PRIVATE src)

enable_testing()
add_test(NAME dart_separation COMMAND test_dart_separation)
add_test(NAME image_index COMMAND test_image_index)

# ---------------------------------------------------------------------------
# Python module (pybind11)
# ---------------------------------------------------------------------------
pybind11_add_module(dartboard_renderer
    src/python_bindings.cpp
    ${RENDERER_CORE_SOURCES}
)

add_dependencies(dartboard_renderer shaders)

target_include_directories(dartboard_renderer PRIVATE src)

target_link_libraries(dartboard_renderer PRIVATE ${RENDERER_CORE_LIBS})

# ---------------------------------------------------------------------------
# Windows: runtime DLLs next to the outputs
# ---------------------------------------------------------------------------
# Imported shared libraries (libjpeg-turbo's turbojpeg.dll) are not on PATH by
# default, and Python does not search PATH for an extension module's
# dependencies at all. Copying them beside the executable and the module is
# what lets both load from a fresh build.
if(WIN32)
    foreach(_target dartboard_gen dartboard_renderer)
        add_custom_command(TARGET ${_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E
                $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:${_target}>>,copy_if_different,true>
                $<TARGET_RUNTIME_DLLS:${_target}> $<TARGET_FILE_DIR:${_target}>
            COMMAND_EXPAND_LISTS)
    endforeach()
endif()
