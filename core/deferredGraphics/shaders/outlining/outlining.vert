#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"

layout(set = 0, binding = 0) uniform GlobalUniformBuffer {
    mat4 view;
    mat4 proj;
} global;

layout(set = 1, binding = 0) uniform LocalUniformBuffer {
    mat4 matrix;
    vec4 constColor;
    vec4 colorFactor;
    vec4 bloomColor;
    vec4 bloomFactor;
    vec4 outliningColor;
    float width;
} local;

layout(set = 2, binding = 0) uniform UBONode {
    mat4 matrix;
    mat4 jointMatrix[MAX_NUM_JOINTS];
} node;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV0;
layout(location = 3) in vec2 inUV1;
layout(location = 4) in vec4 inJoint0;
layout(location = 5) in vec4 inWeight0;
layout(location = 6) in vec3 inTangent;
layout(location = 7) in vec3 inBitangent;

layout(location = 0) out vec4 outColor;

void main() {
    mat4 skinMat =  inJoint0.x != -1.0 && inJoint0.y != -1.0 &&inJoint0.z != -1.0 &&inJoint0.w != -1.0 ?
                    inWeight0.x * node.jointMatrix[int(inJoint0.x)] +
                    inWeight0.y * node.jointMatrix[int(inJoint0.y)] +
                    inWeight0.z * node.jointMatrix[int(inJoint0.z)] +
                    inWeight0.w * node.jointMatrix[int(inJoint0.w)] : mat4(1.0f);
    mat4x4 model = local.matrix * node.matrix * skinMat;

    vec4 position = model * vec4(inPosition.xyz, 1.0);
    vec3 Normal = normalize(vec3(inverse(transpose(model)) * vec4(inNormal, 0.0)));
    position = vec4(position.xyz + Normal * local.width, 1.0);
    outColor = local.outliningColor;

    gl_Position = global.proj * global.view * position;
}
