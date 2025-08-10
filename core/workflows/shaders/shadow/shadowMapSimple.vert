#version 450

#include "../__methods__/defines.glsl"

layout(set = 0, binding = 0) uniform LightBufferObject
{
    mat4 proj;
    mat4 view;
    vec4 lightColor;
    vec4 lightProp;
}light;

layout (set = 1, binding = 0) uniform LocalUniformBuffer
{
    mat4 matrix;
} local;

layout (set = 2, binding = 0) uniform UBONode
{
    mat4 matrix;
    mat4 jointMatrix[MAX_NUM_JOINTS];
} node;

layout(location = 0)	in  vec3 inPosition;
layout(location = 1)	in  vec3 inNormal;
layout(location = 2)	in  vec2 inUV0;

void main()
{
    mat4 model = local.matrix * node.matrix;

    gl_Position = light.proj * light.view * model * vec4(inPosition, 1.0);
}
