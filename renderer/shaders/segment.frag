#version 450

// Per-pixel class and instance ids for dense pretraining targets.
//
// Written to an integer attachment at output resolution with no MSAA, because
// both the main pass's multisample resolve and the kSupersample box filter
// average their inputs. Averaging is right for colour and destroys ids: the
// mean of dart 1 and dart 2 is dart 1.5, and the mean of "flight" and "board"
// is "wire". A separate single-sampled pass is nearest by construction.
//
// The cost of that choice is aliased edges -- a pixel is wholly one class, with
// no partial coverage -- which is the correct trade for a target used as a
// classification label rather than displayed.

layout(push_constant) uniform SegPush {
    mat4 mvp;
    uint segClass;    // dart::SegClass
    uint segInstance; // 0 = not a dart, otherwise dart index + 1
    uint pad0;
    uint pad1;
} pc;

layout(location = 0) out uvec4 outIds;

void main() {
    // b and a are spare; kept so the attachment is a standard 4-component
    // format rather than an R8G8 that some drivers treat as a second-class
    // render target.
    outIds = uvec4(pc.segClass, pc.segInstance, 0u, 255u);
}
