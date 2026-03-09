#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"
#include "../../../workflows/shaders/__methods__/colorFunctions.glsl"
#include "../../../workflows/shaders/__methods__/geometricFunctions.glsl"

layout(push_constant) uniform PC {
    float minAmbientFactor;
} pc;

layout(input_attachment_index = 0, binding = 0) uniform subpassInput inNormalTexture;
layout(input_attachment_index = 1, binding = 1) uniform subpassInput inBaseColorTexture;
layout(input_attachment_index = 2, binding = 2) uniform subpassInput inEmissiveTexture;
layout(input_attachment_index = 3, binding = 3) uniform subpassInput inDepthTexture;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBloom;

void main() {
    const vec4 normal = subpassLoad(inNormalTexture);
    const vec4 baseColorTexture = subpassLoad(inBaseColorTexture);
    const float ao = pc.minAmbientFactor * decodeParameter(0x00ff0000, 16, normal.b) / 255.0f;

    const vec4 emissive = subpassLoad(inEmissiveTexture);

    outColor = vec4(ao * baseColorTexture.rgb, baseColorTexture.a);
    outBloom = emissive;
}
