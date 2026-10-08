#version 450

// Geometry-only vertex stage for the segmentation pass.
//
// Shares the main pass's vertex buffer layout so the same meshes can be drawn
// without a second upload, but reads only the position: this pass produces
// per-pixel class and instance ids, and nothing about lighting, texturing or
// normals affects them.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;

layout(push_constant) uniform SegPush {
    mat4 mvp;
    uint segClass;    // dart::SegClass
    uint segInstance; // 0 = not a dart, otherwise dart index + 1
    uint pad0;
    uint pad1;
} pc;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
}
