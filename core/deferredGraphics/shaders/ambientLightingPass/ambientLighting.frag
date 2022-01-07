#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"
#include "../../../workflows/shaders/__methods__/colorFunctions.glsl"
#include "../../../workflows/shaders/__methods__/geometricFunctions.glsl"

layout(push_constant) uniform PC {
    float minAmbientFactor;
} pc;

layout(input_attachment_index = 0, binding = 0) uniform subpassInput inPositionTexture;
layout(input_attachment_index = 1, binding = 1) uniform subpassInput inNormalTexture;
layout(input_attachment_index = 2, binding = 2) uniform subpassInput inBaseColorTexture;
layout(input_attachment_index = 3, binding = 3) uniform subpassInput inDepthTexture;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBlur;
layout(location = 2) out vec4 outBloom;

vec4 ambient(vec4 baseColorTexture, float minAmbientFactor) {
    vec4 baseColor = SRGBtoLINEAR(baseColorTexture);
    vec3 diffuseColor = minAmbientFactor * baseColor.rgb;
    return vec4(diffuseColor.xyz, baseColorTexture.a);
}

void main() {
    vec4 position = subpassLoad(inPositionTexture);
    vec4 normal = subpassLoad(inNormalTexture);
    vec4 baseColorTexture = subpassLoad(inBaseColorTexture);
    vec4 emissiveTexture = decodeFromFloat(normal.a);

    float ao = decodeParameter(0x000ff0000, 16, position.a) / 255.0f;
    outColor = ao * SRGBtoLINEAR(emissiveTexture) + (checkZeroNormal(normal.xyz) ? SRGBtoLINEAR(baseColorTexture) : ambient(baseColorTexture, pc.minAmbientFactor));
    outBloom = SRGBtoLINEAR(emissiveTexture) + (checkBrightness(outColor) ? outColor : vec4(0.0, 0.0, 0.0, 1.0));
    outBlur = vec4(0.0, 0.0, 0.0, 0.0);
}
