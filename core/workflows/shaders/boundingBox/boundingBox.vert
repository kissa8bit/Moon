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

void main()
{
    vec3 bmin = pushConstants.min.xyz;
    vec3 bmax = pushConstants.max.xyz;
    int joint = int(pushConstants.max.w);

    vec3 vertices[24] = vec3[](
        vec3(bmin.x,bmin.y,bmin.z),
        vec3(bmax.x,bmin.y,bmin.z),

        vec3(bmax.x,bmin.y,bmin.z),
        vec3(bmax.x,bmax.y,bmin.z),

        vec3(bmax.x,bmax.y,bmin.z),
        vec3(bmin.x,bmax.y,bmin.z),

        vec3(bmin.x,bmax.y,bmin.z),
        vec3(bmin.x,bmin.y,bmin.z),

        vec3(bmin.x,bmin.y,bmax.z),
        vec3(bmax.x,bmin.y,bmax.z),

        vec3(bmax.x,bmin.y,bmax.z),
        vec3(bmax.x,bmax.y,bmax.z),

        vec3(bmax.x,bmax.y,bmax.z),
        vec3(bmin.x,bmax.y,bmax.z),

        vec3(bmin.x,bmax.y,bmax.z),
        vec3(bmin.x,bmin.y,bmax.z),

        vec3(bmin.x,bmin.y,bmin.z),
        vec3(bmin.x,bmin.y,bmax.z),

        vec3(bmax.x,bmin.y,bmin.z),
        vec3(bmax.x,bmin.y,bmax.z),

        vec3(bmax.x,bmax.y,bmin.z),
        vec3(bmax.x,bmax.y,bmax.z),

        vec3(bmin.x,bmax.y,bmin.z),
        vec3(bmin.x,bmax.y,bmax.z)
    );

    mat4 skinMat = (joint != -1 ? node.jointMatrix[joint] : mat4(1.0f));
    gl_Position = global.proj * global.view * local.matrix * node.matrix * skinMat * vec4(vertices[gl_VertexIndex], 1.0f);
}
