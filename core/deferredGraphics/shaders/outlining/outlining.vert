#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"

layout(set = 0, binding = 0) uniform GlobalUniformBuffer {
    mat4 view;
    mat4 proj;
} global;

struct Outlining{
    vec4 color;
    float width;
};

layout(set = 2, binding = 0) uniform LocalUniformBuffer {
    mat4 matrix;
    vec4 constColor;
    vec4 colorFactor;
    vec4 bloomColor;
    vec4 bloomFactor;
    Outlining outlining;
} object;

layout(set = 3, binding = 0) uniform UBONode {
    mat4 matrix;
    mat4 jointMatrix[MAX_NUM_JOINTS];
} node;

layout(set = 4, binding = 0) uniform MorphWeightsBuffer {
    uint count;
    vec4 weights[64];
} morphWeights;

layout(set = 6, binding = 0) readonly buffer MorphDeltasBuffer {
    uint morphTargetCount;
    uint vertexCount;
    uint vertexStart;
    uint pad;
    vec4 data[];
} morphDeltas;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV0;
layout(location = 3) in vec2 inUV1;
layout(location = 4) in vec4 inJoint;
layout(location = 5) in vec4 inWeight;
layout(location = 6) in vec3 inTangent;

layout(location = 0) out vec4 outColor;

void main() {
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
    vec3 morphedNormal = inNormal;
    if (morphDeltas.morphTargetCount > 0) {
        uint localIdx = gl_VertexIndex - morphDeltas.vertexStart;
        for (uint i = 0; i < morphDeltas.morphTargetCount; i++) {
            float w = morphWeights.weights[i / 4u][i % 4u];
            if (w != 0.0) {
                uint posIdx = i * morphDeltas.vertexCount + localIdx;
                uint normIdx = morphDeltas.morphTargetCount * morphDeltas.vertexCount + i * morphDeltas.vertexCount + localIdx;
                morphedPosition += w * morphDeltas.data[posIdx].xyz;
                morphedNormal += w * morphDeltas.data[normIdx].xyz;
            }
        }
        morphedNormal = normalize(morphedNormal);
    }

    vec4 position = model * vec4(morphedPosition, 1.0);
    vec3 normal = normalize(vec3(model * vec4(morphedNormal, 0.0)));
    position.xyz += normal * object.outlining.width;

    gl_Position = global.proj * global.view * position;
    outColor = object.outlining.color;
}
