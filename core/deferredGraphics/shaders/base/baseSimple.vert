#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"

layout(set = 0, binding = 0) uniform GlobalBuffer
{
    mat4 view;
    mat4 proj;
} global;

layout (set = 1, binding = 0) uniform ObjectBuffer
{
    mat4 matrix;
    vec4 constColor;
    vec4 colorFactor;
    vec4 bloomColor;
    vec4 bloomFactor;
    vec4 outliningColor;
    float width;
} object;

layout (set = 2, binding = 0) uniform NodeBuffer
{
    mat4 matrix;
    mat4 jointMatrix[MAX_NUM_JOINTS];
} node;

layout(location = 0)	in vec3 inPosition;
layout(location = 1)	in vec3 inNormal;
layout(location = 2)	in vec2 inUV0;

layout(location = 0)	out vec4 outPosition;
layout(location = 1)	out vec3 outNormal;
layout(location = 2)	out vec2 outUV0;
layout(location = 3)	out vec2 outUV1;
layout(location = 4)	out vec3 outTangent;
layout(location = 5)	out vec3 outBitangent;

void main()
{
    outUV0 = inUV0;
    outUV1 = inUV0;

    mat4x4 model = object.matrix * node.matrix;

    outPosition     = model * vec4(inPosition,	1.0);
    outNormal	    = normalize(vec3(model * vec4(inNormal, 0.0)));
    outTangent	    = vec3(0.0);
    outBitangent    = vec3(0.0);

    gl_Position = global.proj * global.view * outPosition;
}
