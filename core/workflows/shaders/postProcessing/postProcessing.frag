#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"

layout(constant_id = 0) const int bloomCount = 1;

layout(set = 0, binding = 0) uniform sampler2D Sampler;
layout(set = 0, binding = 1) uniform sampler2D blurSampler;
layout(set = 0, binding = 2) uniform sampler2D bloomSampler;
layout(set = 0, binding = 3) uniform sampler2D ssaoSampler;
layout(set = 0, binding = 4) uniform sampler2D bbSampler;

//vec4 colorBloomFactor[bloomCount] = vec4[](
//    vec4(1.0f,0.0f,0.0f,1.0f),
//    vec4(0.0f,0.0f,1.0f,1.0f),
//    vec4(0.0f,1.0f,0.0f,1.0f),
//    vec4(1.0f,0.0f,0.0f,1.0f),
//    vec4(0.0f,0.0f,1.0f,1.0f),
//    vec4(0.0f,1.0f,0.0f,1.0f),
//    vec4(1.0f,0.0f,0.0f,1.0f),
//    vec4(1.0f,1.0f,1.0f,1.0f)
//);

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    float blitFactor;
} pc;

vec4 blur(sampler2D Sampler, vec2 TexCoord) {
    float twoPi = 2 * pi;

    float Directions = 16.0; // BLUR DIRECTIONS (Default 16.0 - More is better but slower)
    float Quality = 3.0; // BLUR QUALITY (Default 4.0 - More is better but slower)
    float Size = 8.0; // BLUR SIZE (Radius)

    vec2 Radius = Size / textureSize(blurSampler, 0);

    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = TexCoord;
    // Pixel colour
    vec4 Color = texture(Sampler, uv);

    // Blur calculations
    for(float d = 0.0; d < twoPi; d += twoPi / Directions) {
        for(float i = 1.0 / Quality; i <= 1.0; i += 1.0 / Quality) {
            Color += texture(Sampler, uv + vec2(cos(d), sin(d)) * Radius * i);
        }
    }

    // Output to screen
    return Color /= Quality * Directions - 15.0;
}

vec4 radialBlur(sampler2D Sampler, vec2 TexCoord) {
    int Samples = 128;
    float Intensity = 0.125, Decay = 0.96875;
    vec2 Direction = vec2(0.5) - TexCoord;
    Direction /= Samples;
    vec3 Color = texture(Sampler, TexCoord).xyz;

    for(int Sample = 0; Sample < Samples; Sample++) {
        Color += texture(Sampler, TexCoord).xyz * Intensity;
        Intensity *= Decay;
        TexCoord += Direction;
    }

    return vec4(Color, 1.0);
}

void main() {
    outColor = vec4(0.0, 0.0, 0.0, 1.0);

    vec4 baseColor = texture(Sampler, fragTexCoord);

    outColor += dot(baseColor,baseColor) > 0.0f ? baseColor : texture(blurSampler, fragTexCoord);
    outColor += texture(bloomSampler, fragTexCoord);
    //outColor += texture(sslrSampler,fragTexCoord);
    outColor += texture(bbSampler,fragTexCoord);
    outColor += vec4(texture(ssaoSampler,fragTexCoord).xyz,0.0f);
}
