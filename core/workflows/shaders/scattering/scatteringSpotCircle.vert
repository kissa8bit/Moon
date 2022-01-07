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
layout(location = 1)	out vec4 glPosition;
layout(location = 2)	out mat4 projview;

vec3 vertex[5];
int index[18] = int[](
        0,4,1,
        0,1,2,
        0,2,3,
        0,3,4,
        4,2,1,
        2,4,3);

void main()
{
    projview	= global.proj * global.view;
    eyePosition = vec4(viewPosition(global.view), 1.0f);

    vec3 u =   normalize(vec3(light.view[0][0],light.view[1][0],light.view[2][0]));
    vec3 v =   normalize(vec3(light.view[0][1],light.view[1][1],light.view[2][1]));
    vec3 n = - normalize(vec3(light.view[0][2],light.view[1][2],light.view[2][2]));

    float far = light.proj[3][2]/(light.proj[2][2] + 1.0f);
    float h = - far/light.proj[1][1];
    float w = light.proj[1][1]/light.proj[0][0]*h;

    vec3 lightPosition = viewPosition(light.view);
    vertex[0] = lightPosition;
    vertex[1] = lightPosition + far*n + w*u + h*v;
    vertex[2] = lightPosition + far*n + w*u - h*v;
    vertex[3] = lightPosition + far*n - w*u - h*v;
    vertex[4] = lightPosition + far*n - w*u + h*v;

    glPosition = vec4(vertex[index[gl_VertexIndex]],1.0f);
    gl_Position = projview * glPosition;
}
