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

layout(set = 1, binding = 0) uniform LocalUniformBuffer {
    mat4 matrix;
    vec4 constColor;
    vec4 colorFactor;
    vec4 bloomColor;
    vec4 bloomFactor;
    Outlining outlining;
} object;

layout(set = 2, binding = 0) uniform UBONode {
    mat4 matrix;
    mat4 jointMatrix[MAX_NUM_JOINTS];
} node;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV0;

layout(location = 0) out vec4 outColor;

void main() {
    mat4x4 model = object.matrix * node.matrix;

    vec4 position = model * vec4(inPosition, 1.0);
    vec3 normal = normalize(vec3(model * vec4(inNormal, 0.0)));
    position.xyz += normal * object.outlining.width;

    gl_Position = global.proj * global.view * position;
    outColor = object.outlining.color;
}
