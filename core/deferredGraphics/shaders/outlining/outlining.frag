#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"
#include "../../../workflows/shaders/__methods__/colorFunctions.glsl"

layout(set = 5, binding = 0) uniform sampler2D baseColorTexture;
layout(set = 5, binding = 1) uniform sampler2D metallicRoughnessTexture;
layout(set = 5, binding = 2) uniform sampler2D normalTexture;
layout(set = 5, binding = 3) uniform sampler2D occlusionTexture;
layout(set = 5, binding = 4) uniform sampler2D emissiveTexture;

layout(location = 0) in vec4 color;

layout(location = 0) out vec4 outNormal;
layout(location = 1) out vec4 outBaseColor;
layout(location = 2) out vec4 outEmissiveColor;

void main() {
    uint outlining = uint(1);
    float params = uintBitsToFloat(outlining << 24);

    outNormal = vec4(0.0, 0.0, params, 0.0);
    outBaseColor = vec4(color.xyz, 1.0);
    outEmissiveColor = vec4((1.0 - color.a) * color.xyz, 1.0);
}
