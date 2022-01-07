#version 450

#include "../__methods__/defines.glsl"
#include "../__methods__/colorFunctions.glsl"

layout(set = 1, binding = 1)	uniform samplerCube samplerCubeMap;

layout(location = 0) in vec3 inUVW;
layout(location = 1) in vec4 constColor;
layout(location = 2) in vec4 colorFactor;

layout(location = 0) out vec4 outBaseColor;
layout(location = 1) out vec4 outBloomColor;

void main()
{
    outBaseColor = colorFactor * texture(samplerCubeMap, inUVW) + constColor;
    outBloomColor = (checkBrightness(outBaseColor)) ? outBaseColor : vec4(0.0f);
}
