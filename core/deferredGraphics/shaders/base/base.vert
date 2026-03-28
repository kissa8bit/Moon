#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"

layout(set = 0, binding = 0) uniform GlobalBuffer
{
    mat4 view;
    mat4 proj;
} global;

layout (set = 2, binding = 0) uniform ObjectBuffer
{
    mat4 matrix;
    vec4 constColor;
    vec4 colorFactor;
    vec4 bloomColor;
    vec4 bloomFactor;
    vec4 outliningColor;
    float width;
} object;

layout (set = 3, binding = 0) uniform NodeBuffer
{
    mat4 matrix;
    mat4 jointMatrix[MAX_NUM_JOINTS];
} node;

layout (set = 4, binding = 0) uniform MorphWeightsBuffer
{
    uint count;
    vec4 weights[64];
} morphWeights;

layout (set = 6, binding = 0) readonly buffer MorphDeltasBuffer
{
    uint morphTargetCount;
    uint vertexCount;
    uint vertexStart;
    uint pad;
    vec4 data[]; // posDeltas[count*vertexCount] then normDeltas[count*vertexCount]
} morphDeltas;

layout(location = 0)	in  vec3 inPosition;
layout(location = 1)	in  vec3 inNormal;
layout(location = 2)	in  vec2 inUV0;
layout(location = 3)	in  vec2 inUV1;
layout(location = 4)	in  vec4 inJoint;
layout(location = 5)	in  vec4 inWeight;
layout(location = 6)	in  vec4 inTangent;

layout(location = 0)	out vec4 outPosition;
layout(location = 1)	out vec3 outNormal;
layout(location = 2)	out vec2 outUV0;
layout(location = 3)	out vec2 outUV1;
layout(location = 4)	out vec4 outTangent;
layout(location = 5)	flat out vec3 outEyePos;

void main()
{
    outUV0 = inUV0;
    outUV1 = inUV1;

    mat4 skinMat = mat4(1.0f);
    if(inJoint.x != -1.0 && inJoint.y != -1.0 && inJoint.z != -1.0 && inJoint.w != -1.0)
    {
        skinMat =
            inWeight.x * node.jointMatrix[int(inJoint.x)] +
            inWeight.y * node.jointMatrix[int(inJoint.y)] +
            inWeight.z * node.jointMatrix[int(inJoint.z)] +
            inWeight.w * node.jointMatrix[int(inJoint.w)];
    }

    mat4x4 model = object.matrix * node.matrix * skinMat;

    vec3 morphedPosition = inPosition;
    vec3 morphedNormal   = inNormal;
    if (morphDeltas.morphTargetCount > 0) {
        uint localIdx = gl_VertexIndex - morphDeltas.vertexStart;
        for (uint i = 0; i < morphDeltas.morphTargetCount; i++) {
            float w = morphWeights.weights[i / 4u][i % 4u];
            if (w != 0.0) {
                uint posIdx  = i * morphDeltas.vertexCount + localIdx;
                uint normIdx = morphDeltas.morphTargetCount * morphDeltas.vertexCount + i * morphDeltas.vertexCount + localIdx;
                morphedPosition += w * morphDeltas.data[posIdx].xyz;
                morphedNormal   += w * morphDeltas.data[normIdx].xyz;
            }
        }
        morphedNormal = normalize(morphedNormal);
    }

    outPosition  = model * vec4(morphedPosition, 1.0);

    mat3 normalMatrix = mat3(model);
    outNormal	 = normalize(normalMatrix * morphedNormal);
    outTangent	 = vec4(normalize(normalMatrix * inTangent.xyz), inTangent.w);

    mat3 viewRot = transpose(mat3(global.view));
    outEyePos    = -viewRot * global.view[3].xyz;

    gl_Position = global.proj * global.view * outPosition;
}
