#version 450

#include "../__methods__/defines.glsl"
#include "../__methods__/geometricFunctions.glsl"
#include "scatteringBase.glsl"

layout(location = 0)	in vec4 eyePosition;
layout(location = 1)	in vec4 fragPosition;
layout(location = 2)	in mat4 projview;

layout(set = 0, binding = 1) uniform sampler2D inDepthTexture;
layout(set = 1, binding = 0) uniform sampler2D shadowMap;
layout(set = 2, binding = 0) uniform LightBufferObject
{
    mat4 proj;
    mat4 view;
    vec4 color;
    vec4 prop;
} light;
layout(set = 2, binding = 1) uniform sampler2D lightTexture;

layout (push_constant) uniform PC
{
    int width;
    int height;
    float density;
    int steps;
}pc;

layout(location = 0) out vec4 outScattering;

void main()
{
    float depthMap = texture(inDepthTexture, vec2(gl_FragCoord.x / pc.width, gl_FragCoord.y / pc.height)).r;

    outScattering = LightScattering(
            pc.steps,
            pc.density,
            light.view,
            light.proj,
            light.color,
            projview,
            eyePosition,
            fragPosition,
            lightTexture,
            shadowMap,
            depthMap,
            light.prop,
            SPOT_LIGHTING_TYPE_CIRCLE);
}
