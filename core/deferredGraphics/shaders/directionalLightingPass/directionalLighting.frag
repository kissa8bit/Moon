#version 450

#include "../../../workflows/shaders/__methods__/defines.glsl"
#include "../../../workflows/shaders/__methods__/colorFunctions.glsl"
#include "../../../workflows/shaders/__methods__/lightDrop.glsl"
#include "../../../workflows/shaders/__methods__/pbr.glsl"
#include "../../../workflows/shaders/__methods__/shadow.glsl"
#include "../../../workflows/shaders/__methods__/geometricFunctions.glsl"

layout(location = 0) in vec4 eyePosition;

layout(set = 1, input_attachment_index = 0, binding = 0) uniform subpassInput inNormalTexture;
layout(set = 1, input_attachment_index = 1, binding = 1) uniform subpassInput inBaseColorTexture;
layout(set = 1, input_attachment_index = 2, binding = 2) uniform subpassInput inEmissiveColorTexture;
layout(set = 1, input_attachment_index = 3, binding = 3) uniform subpassInput inDepthTexture;

layout(set = 0, binding = 0) uniform GlobalUniformBuffer {
    mat4 view;
    mat4 proj;
    mat4 invViewProj;
    vec2 viewport;
} global;

layout(set = 2, binding = 0) uniform sampler2D shadowMap;

layout(set = 3, binding = 0) uniform LightBufferObject {
    mat4 proj;
    mat4 view;
    vec4 color;
    vec4 prop;
} light;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outBloom;

void main() {
    const vec4 normal   = subpassLoad(inNormalTexture);
    const vec4 color    = subpassLoad(inBaseColorTexture);
    const vec4 emissive = subpassLoad(inEmissiveColorTexture);
    const float depth   = subpassLoad(inDepthTexture).x;
    
    outBloom = emissive;

    const bool outlining = uint((0xff000000 & floatBitsToUint(normal.b)) >> 24) == 1;
    if (outlining) {
        outColor = color;
        return;
    }

    const vec3 decodedNormal = decodeSphericalNormal(normal.xy);
    if (checkZeroNormal(decodedNormal)) {
        outColor = vec4(0.0);
        return;
    }

    const vec2 screenUV = gl_FragCoord.xy / global.viewport;
    const vec3 worldPos = reconstructPosition(global.invViewProj, screenUV, depth);

    // Transform fragment to light (shadow camera) space and project
    const vec4 localPosition = light.view * vec4(worldPos, 1.0);
    const vec3 ndcCoords = normalizedCoords(light.proj, localPosition);

    // Only sample shadow map when the fragment is inside the ortho frustum
    if (ndcCoords.x < 0.0 || ndcCoords.x > 1.0 ||
        ndcCoords.y < 0.0 || ndcCoords.y > 1.0 ||
        ndcCoords.z < 0.0 || ndcCoords.z > 1.0) 
    {
        outColor = vec4(0.0);
        return;
    }
    
    float shadowFactor = calcShadowFactor(light.proj, shadowMap, ndcCoords);

    // Light direction is the forward (-Z) axis of the light's view space,
    // expressed in world space. Simulate a point at infinity so pbr() gets
    // the correct constant direction for all fragments.
    const vec3 lightDir = -getNDirection(light.view);
    const vec3 lightPosition = worldPos + lightDir;
    
    const float lightDropFactor = light.prop.w;
    const float lightDrop = lightDrop(max(lightDropFactor, 0.01) * max(-localPosition.z, 0.01));

    const float lightPowerFactor = light.prop.z;
    const float lightPower = shadowFactor * lightPowerFactor / lightDrop;
    const vec4 pbrColor = pbr(worldPos, decodedNormal, normal.b, color, eyePosition, light.color, lightPosition);

    outColor = vec4(lightPower * pbrColor.xyz, pbrColor.a);
}
