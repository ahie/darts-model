#version 460

// Per-vertex attributes
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;

// Push constants (128 bytes)
layout(push_constant) uniform PushConstants {
    mat4 mvp;            // 64 bytes
    vec4 modelCol0;      // model matrix column 0 (x-axis), w = tx
    vec4 modelCol1;      // model matrix column 1 (y-axis), w = ty
    vec4 modelCol2;      // model matrix column 2 (z-axis), w = tz
    uint drawMode;       // bit31=useTexture, bit30=hasNormalMap, bit29=board face,
                         // bit28=translucent (alpha in bits 6:0),
                         // bits22:15=metallic, bits14:7=roughness
    float colorR;
    float colorG;
    float colorB;
} pc;

// Outputs to fragment shader
layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;
layout(location = 3) flat out uint fragDrawMode;
layout(location = 4) flat out vec3 fragColor;
layout(location = 5) out vec3 fragTangent;
layout(location = 6) out float fragBitangentSign;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);

    // Reconstruct model matrix for world-space transforms
    mat3 modelRot = mat3(
        pc.modelCol0.xyz,
        pc.modelCol1.xyz,
        pc.modelCol2.xyz
    );
    vec3 modelTranslation = vec3(pc.modelCol0.w, pc.modelCol1.w, pc.modelCol2.w);

    fragWorldPos = modelRot * inPosition + modelTranslation;
    fragNormal = normalize(modelRot * inNormal);
    fragUV = inUV;
    fragDrawMode = pc.drawMode;
    fragColor = vec3(pc.colorR, pc.colorG, pc.colorB);
    fragTangent = normalize(modelRot * inTangent.xyz);
    fragBitangentSign = inTangent.w;
}
