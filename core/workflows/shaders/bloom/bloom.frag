#version 450

#include "../__methods__/defines.glsl"

layout(constant_id = 0) const int bloomCount = 1;

layout(set = 0, binding = 0) uniform sampler2D bloomSampler[bloomCount];

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
    float dx;
    float dy;
    float blitFactor;
} pc;

vec4 bloom() {
    float blitFactor = pc.blitFactor;
    vec4 bloomColor = vec4(0.0);
    float invBlitFactor = 1.0 / blitFactor;
    for(int i = 0; i < bloomCount; i++) {
        vec2 coord = fragTexCoord * invBlitFactor;
        bloomColor += /*colorBloomFactor[i] * exp(0.01*i*i) **/ texture(bloomSampler[i], coord);
        invBlitFactor /= blitFactor;
    }

    return bloomColor;
}

void main() {
    outColor = vec4(0.0, 0.0, 0.0, 1.0);
    outColor += bloom();
}
