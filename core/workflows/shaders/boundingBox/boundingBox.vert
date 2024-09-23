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
    float jointCount;
    mat4 matrix;
    mat4 jointMatrix[MAX_NUM_JOINTS];
} node;

layout (push_constant) uniform PushConstants{
    vec3 min;
    vec3 max;
} pushConstants;

layout(location = 0)	in  vec3 inPosition;
layout(location = 1)	in  vec3 inNormal;
layout(location = 2)	in  vec2 inUV0;
layout(location = 3)	in  vec2 inUV1;
layout(location = 4)	in  vec4 inJoint0;
layout(location = 5)	in  vec4 inWeight0;
layout(location = 6)	in  vec3 inTangent;
layout(location = 7)	in  vec3 inBitangent;

vec3 min = pushConstants.min;
vec3 max = pushConstants.max;

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
    mat4 skinMat = node.jointCount > 0.0 ?
        inWeight0.x * node.jointMatrix[int(inJoint0.x)] +
        inWeight0.y * node.jointMatrix[int(inJoint0.y)] +
        inWeight0.z * node.jointMatrix[int(inJoint0.z)] +
        inWeight0.w * node.jointMatrix[int(inJoint0.w)] : mat4(1.0f);

    mat4x4 model = local.matrix * node.matrix * skinMat;

    vec3 Position = vertex[gl_VertexIndex];

    gl_Position = global.proj * global.view * model * vec4(Position,1.0f);
}
