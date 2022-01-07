#version 450

#include "../__methods__/defines.glsl"
#include "../__methods__/geometricFunctions.glsl"

layout (set = 0, binding = 0) uniform UniformBuffer
{
    mat4 view;
    mat4 proj;
} camera;

layout (set = 1, binding = 0) uniform LocalUniformBuffer
{
    mat4 matrix;
    vec4 constColor;
    vec4 colorFactor;
    vec4 bloomColor;
    vec4 bloomFactor;
} local;

layout(location = 0)	out vec3 outUVW;
layout(location = 1)	out vec4 constColor;
layout(location = 2)	out vec4 colorFactor;

vec3 vertex[36] = vec3[](
    vec3(-1.0f,-1.0f,-1.0f),
    vec3(-1.0f,-1.0f, 1.0f),
    vec3(-1.0f, 1.0f, 1.0f),
    vec3( 1.0f, 1.0f,-1.0f),
    vec3(-1.0f,-1.0f,-1.0f),
    vec3(-1.0f, 1.0f,-1.0f),
    vec3( 1.0f,-1.0f, 1.0f),
    vec3(-1.0f,-1.0f,-1.0f),
    vec3( 1.0f,-1.0f,-1.0f),
    vec3( 1.0f, 1.0f,-1.0f),
    vec3( 1.0f,-1.0f,-1.0f),
    vec3(-1.0f,-1.0f,-1.0f),
    vec3(-1.0f,-1.0f,-1.0f),
    vec3(-1.0f, 1.0f, 1.0f),
    vec3(-1.0f, 1.0f,-1.0f),
    vec3( 1.0f,-1.0f, 1.0f),
    vec3(-1.0f,-1.0f, 1.0f),
    vec3(-1.0f,-1.0f,-1.0f),
    vec3(-1.0f, 1.0f, 1.0f),
    vec3(-1.0f,-1.0f, 1.0f),
    vec3( 1.0f,-1.0f, 1.0f),
    vec3( 1.0f, 1.0f, 1.0f),
    vec3( 1.0f,-1.0f,-1.0f),
    vec3( 1.0f, 1.0f,-1.0f),
    vec3( 1.0f,-1.0f,-1.0f),
    vec3( 1.0f, 1.0f, 1.0f),
    vec3( 1.0f,-1.0f, 1.0f),
    vec3( 1.0f, 1.0f, 1.0f),
    vec3( 1.0f, 1.0f,-1.0f),
    vec3(-1.0f, 1.0f,-1.0f),
    vec3( 1.0f, 1.0f, 1.0f),
    vec3(-1.0f, 1.0f,-1.0f),
    vec3(-1.0f, 1.0f, 1.0f),
    vec3( 1.0f, 1.0f, 1.0f),
    vec3(-1.0f, 1.0f, 1.0f),
    vec3( 1.0f,-1.0f, 1.0f)
);

void main()
{
    constColor = local.constColor;
    colorFactor = local.colorFactor;

    mat4x4 cameraModel = mat4x4(1.0f);
    vec3 eyePosition = viewPosition(camera.view);
    cameraModel[3][0] = eyePosition.x;
    cameraModel[3][1] = eyePosition.y;
    cameraModel[3][2] = eyePosition.z;

    vec3 Position = vertex[gl_VertexIndex];
    outUVW = vec4(vec4(Position,1.0f)).xyz;
    gl_Position = camera.proj * camera.view * cameraModel * local.matrix * vec4(Position,1.0f);
}
