#version 450

#include "../__methods__/defines.glsl"
#include "../__methods__/geometricFunctions.glsl"

layout(set = 0, binding = 0) uniform GlobalUniformBuffer
{
    mat4 view;
    mat4 proj;
} global;

layout(set = 2, binding = 0) uniform LightBufferObject
{
    mat4 proj;
    mat4 view;
    vec4 color;
    vec4 prop;
}light;

layout(location = 0)	out vec4 eyePosition;
layout(location = 1)	out vec4 fragPosition;
layout(location = 2)	out mat4 projview;

vec3 vertex[5];
PYRAMID_INDICES

void main()
{
    const int index = pyramid_indices[gl_VertexIndex];
    const vec3 pos = getPyramidVertex(light.proj, light.view, index);
 
    eyePosition = vec4(viewPosition(global.view), 1.0f);   
    fragPosition = vec4(pos, 1.0f);
    projview = global.proj * global.view;

    gl_Position = projview * fragPosition;
}
