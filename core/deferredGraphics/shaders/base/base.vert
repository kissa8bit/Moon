#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"

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
    vec4 outliningColor;
    float width;
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
layout(location = 7)	in  vec3 inBitangent;

layout(location = 0)	out vec4 outPosition;
layout(location = 1)	out vec3 outNormal;
layout(location = 2)	out vec2 outUV0;
layout(location = 3)	out vec2 outUV1;
layout(location = 4)	out vec3 outTangent;
layout(location = 5)	out vec3 outBitangent;
layout(location = 6)	out vec3 glPosition;

void main()
{
    outUV0 = inUV0;
    outUV1 = inUV1;

    mat4 skinMat = inJoint0.x != -1.0 && inJoint0.y != -1.0 &&inJoint0.z != -1.0 &&inJoint0.w != -1.0 ?
        inWeight0.x * node.jointMatrix[int(inJoint0.x)] +
        inWeight0.y * node.jointMatrix[int(inJoint0.y)] +
        inWeight0.z * node.jointMatrix[int(inJoint0.z)] +
        inWeight0.w * node.jointMatrix[int(inJoint0.w)] : mat4(1.0f);

    mat4x4 model = local.matrix * node.matrix * skinMat;

    outPosition  = model * vec4(inPosition,	1.0);
    outNormal	 = normalize(vec3(transpose(inverse(model)) * vec4(inNormal, 0.0)));
    outTangent	 = normalize(vec3(model * vec4(inTangent, 0.0)));
    outBitangent = normalize(vec3(model * vec4(inBitangent,	0.0)));

    gl_Position = global.proj * global.view * outPosition;
    glPosition = vec3(gl_Position.xy / gl_Position.w * 0.5f + 0.5f, gl_Position.z / gl_Position.w);
}
