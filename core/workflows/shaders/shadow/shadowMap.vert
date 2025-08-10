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
layout(location = 3)	in  vec2 inUV1;
layout(location = 4)	in  vec4 inJoint0;
layout(location = 5)	in  vec4 inWeight0;
layout(location = 6)	in  vec3 inTangent;

void main()
{
    mat4 model = local.matrix * node.matrix;

    mat4 skinMat = inJoint0.x != -1.0 && inJoint0.y != -1.0 &&inJoint0.z != -1.0 &&inJoint0.w != -1.0 ?
        inWeight0.x * node.jointMatrix[int(inJoint0.x)] +
        inWeight0.y * node.jointMatrix[int(inJoint0.y)] +
        inWeight0.z * node.jointMatrix[int(inJoint0.z)] +
        inWeight0.w * node.jointMatrix[int(inJoint0.w)] : mat4(1.0f);

    gl_Position = light.proj * light.view * model * skinMat* vec4(inPosition, 1.0);
}
