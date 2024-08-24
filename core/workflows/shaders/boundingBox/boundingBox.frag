#version 450

#include "../__methods__/defines.glsl"

layout(set = 0, binding = 1) uniform sampler2D Sampler;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = vec4(0.0f, 1.0f, 0.0f, 1.0f);
}