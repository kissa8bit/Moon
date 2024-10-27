#version 450

#include "../__methods__/defines.glsl"
#include "../__methods__/geometricFunctions.glsl"

layout(set = 0, binding = 0) uniform GlobalUniformBuffer
{
    mat4 view;
    mat4 proj;
} global;

layout (set = 1, binding = 0) uniform LocalUniformBuffer
{
    mat4 matrix;
    vec4 constColor;
    vec4 colorFactor;
    vec4 bloomColor;
    vec4 bloomFactor;
} local;

layout (set = 2, binding = 0) uniform UBONode
{
    mat4 matrix;
    mat4 jointMatrix[MAX_NUM_JOINTS];
} node;

layout (push_constant) uniform PushConstants{
    vec4 min;
    vec4 max;
} pushConstants;

vec3 min = pushConstants.min.xyz;
vec3 max = pushConstants.max.xyz;
int joint = int(pushConstants.max.w);

vec3 vertex[24] = vec3[](
    vec3(min.x,min.y,min.z),
    vec3(max.x,min.y,min.z),

    vec3(max.x,min.y,min.z),
    vec3(max.x,max.y,min.z),

    vec3(max.x,max.y,min.z),
    vec3(min.x,max.y,min.z),

    vec3(min.x,max.y,min.z),
    vec3(min.x,min.y,min.z),

    vec3(min.x,min.y,max.z),
    vec3(max.x,min.y,max.z),

    vec3(max.x,min.y,max.z),
    vec3(max.x,max.y,max.z),

    vec3(max.x,max.y,max.z),
    vec3(min.x,max.y,max.z),

    vec3(min.x,max.y,max.z),
    vec3(min.x,min.y,max.z),

    vec3(min.x,min.y,min.z),
    vec3(min.x,min.y,max.z),

    vec3(max.x,min.y,min.z),
    vec3(max.x,min.y,max.z),

    vec3(max.x,max.y,min.z),
    vec3(max.x,max.y,max.z),

    vec3(min.x,max.y,min.z),
    vec3(min.x,max.y,max.z)
);

void main()
{
    mat4 skinMat = (joint != -1 ? node.jointMatrix[joint] : mat4(1.0f));
    gl_Position = global.proj * global.view * local.matrix * node.matrix * skinMat * vec4(vertex[gl_VertexIndex], 1.0f);
}
