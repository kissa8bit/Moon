#version 450

#include "../__methods__/defines.glsl"
#include "../__methods__/math.glsl"
#include "blur.glsl"

layout(set = 0, binding = 0) uniform sampler2D color;
layout(set = 0, binding = 1) uniform sampler2D depth;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    float depth;
} pc;

void main() {
    outColor = blur(color, depth, fragTexCoord, vec2(0.0, 1.0), pc.depth);
}
