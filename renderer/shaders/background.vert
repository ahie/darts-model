#version 450

layout(location = 0) out vec2 fragUV;

void main() {
    // Fullscreen triangle: 3 vertices cover the entire screen
    //   v0 = (-1, -1), v1 = (3, -1), v2 = (-1, 3)
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    fragUV = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 1.0, 1.0); // z = 1.0 (far plane)
}
