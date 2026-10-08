#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D bgTexture;

layout(push_constant) uniform BgPushConstants {
    vec4 colorA;    // rgb (linear) + w = mode (0 = Perlin, 1 = texture)
    vec4 colorB;    // rgb (linear) + unused
    vec4 params;    // x = scale, y = offsetX, z = offsetY, w = intensity
} bg;

// --- Perlin noise helpers (sin-based hash, no textures needed) ---

vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

float perlinNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    // Quintic interpolation curve
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    float a = dot(hash2(i + vec2(0.0, 0.0)), f - vec2(0.0, 0.0));
    float b = dot(hash2(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0));
    float c = dot(hash2(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0));
    float d = dot(hash2(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 6; i++) {
        value += amplitude * perlinNoise(p);
        p *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

void main() {
    if (bg.colorA.w > 0.5) {
        // Texture mode. The photo is an SRGB texture, so the sample arrives
        // linear, and the SRGB colour attachment re-encodes it on store with
        // the same exact curve, so the photo comes out without a tone shift.
        outColor = vec4(texture(bgTexture, fragUV).rgb, 1.0);
    } else {
        // Perlin noise mode
        vec2 uv = fragUV * bg.params.x + vec2(bg.params.y, bg.params.z);
        float intensity = bg.params.w;
        float n = fbm(uv) * 0.5 + 0.5; // remap [-1,1] -> [0,1]

        // Sharpen: push values towards 0 or 1 to create harder edges
        n = smoothstep(0.3, 0.7, n);

        n = mix(0.5, n, intensity);     // intensity controls contrast (0=flat, 1=full)
        // colorA/colorB are LINEAR: the attachment encodes to sRGB on store.
        vec3 color = mix(bg.colorA.rgb, bg.colorB.rgb, n);
        outColor = vec4(color, 1.0);
    }
}
